import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

def check_nan(x):
    return torch.isnan(x).int().sum()

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm + 1e-10)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

def euclidean_sim(im, s):
    # im: (B, N, T)

    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1), s.size(2))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1), s.size(2)))
    score = -YmX.pow(2).sum(2).sqrt().mean(2).t()
    return score

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.measure = measure
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s, mask=None):
        # compute image-sentence score matrix

        B = im.size(0)
        scores = self.sim(im, s)
        
        diagonal = scores.diag().view(B, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        # clear diagonals
        if mask is None:
            mask = torch.eye(scores.size(0)) > .5
        else:
            mask = mask > .5
            
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features

class Video_multilevel_encoding(nn.Module):
    def __init__(self, opt, visual_feat_dim=None):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.concate
        if visual_feat_dim is None:
            self.visual_feat_dim = opt.visual_feat_dim
        else:
            self.visual_feat_dim = visual_feat_dim

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(self.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.visual_kernel_sizes
            ])

        # visual mapping
        self.visual_mapping_layers = opt.visual_mapping_layers
        self.visual_mapping_layers[0] = self.visual_feat_dim + opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
        self.visual_mapping = MFC(self.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, videos):
        """Extract video feature vectors."""

        videos, videos_origin, lengths, vidoes_mask = videos
        
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)


class Text_multilevel_encoding(nn.Module):
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.text_norm = opt.text_norm
        self.concate = opt.concate

        # text bidirectional rnn encoder
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # text 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.text_kernel_sizes
            ])

        # text mapping
        self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, videos):
        """Extract video feature vectors."""

        videos, videos_origin, lengths, vidoes_mask = videos
        
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Text_multilevel_encoding, self).load_state_dict(new_state)
        

class COEMB_VRNN_TRNN(object):
    
    def __init__(self, opt):
        self.grad_clip = opt.grad_clip
        self.img_enc = Video_multilevel_encoding(opt)
        self.txt_enc = Text_multilevel_encoding(opt)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, videos, targets, volatile=False):
        """Compute the image and caption embeddings
        """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = videos
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)
        
        
        # Set mini-batch dataset
        
        captions, mean_cap, lengths, cap_masks = targets
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if mean_cap is not None:
            mean_cap = Variable(mean_cap, volatile=volatile)
            if torch.cuda.is_available():
                mean_cap = mean_cap.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        text_data = (captions, mean_cap, lengths, cap_masks)

        # Forward
        img_emb = self.img_enc(videos_data)
        cap_emb = self.txt_enc(text_data)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, mask=None, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, mask=mask)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, videos, captions, ids, vids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(videos, captions)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
        
        