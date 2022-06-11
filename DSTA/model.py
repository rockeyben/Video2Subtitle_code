import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import torch.nn.functional as F
from end2end_dp_batch import End2EndDP
import time

def check_nan(x):
    return torch.isnan(x).int().sum()

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm + 1e-10)
    return X

def l2norm_3D(X):
    """L2-normalize columns of X
    """
    norm = (torch.pow(X, 2).sum(dim=1, keepdim=True)).sqrt()
    #print(norm.size())
    X = torch.div(X, norm)
    return X

def l2norm_A2(X):
    """L2-normalize columns of X
    """
    norm = (torch.pow(X, 2).sum(dim=2, keepdim=True) + 1e-10).sqrt()
    #print(norm.size())
    X = torch.div(X, norm + 1e-10)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)



class EncoderImageSDTW_v2(nn.Module):

    def __init__(self, img_dim, embed_size, norm_method='bn'):
        super(EncoderImageSDTW_v2, self).__init__()
        self.embed_size = embed_size
        self.non_linear = False

        self.conv = nn.Conv1d(img_dim, embed_size, 3)
        self.relu1 = nn.ReLU()
        
        self.pool1 = nn.AvgPool1d(3, stride=2)
        self.conv2 = nn.Conv1d(embed_size, embed_size, 3)
        
        self.pool2 = nn.AvgPool1d(3, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(embed_size, embed_size, 1)

        self.norm_method = norm_method

        if self.norm_method == 'bn':
            self.bn1 = nn.BatchNorm1d(embed_size)
            self.bn2 = nn.BatchNorm1d(embed_size)
        elif self.norm_method == 'none':
            self.bn1 = nn.Dropout(p=0.2)
            self.bn2 = nn.Dropout(p=0.2)

        self.init_weights()


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)


    def forward(self, videos):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        videos, videos_origin, lengths, vidoes_mask = videos

        x = self.conv(videos.permute(0,2,1)) # (B, N, T)
        
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = x.permute(0,2,1)
        
        return x

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = own_state.copy()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageSDTW_v2, self).load_state_dict(new_state)



def end2end_dp_v2(im, s, len_a, len_b, criterion):
    B = im.shape[0]
    M = im.shape[1]
    N = s.shape[1]
    C = im.shape[2]
    

    D = torch.zeros((B*B, M, N))
    cur_len_a = torch.zeros((B*B))
    cur_len_b = torch.zeros((B*B))

    for i in range(B):
        for j in range(B):
            cur_len_a[B*i+j] = len_a[i]
            cur_len_b[B*i+j] = len_b[j]
            D[B*i+j, :, :] = 1 - F.normalize(im[i]).mm(F.normalize(s[j]).t())
    
    score = criterion(D, cur_len_a.int(), cur_len_b.int())
    score = score / 100

    return -score.view((B, B)).cuda()

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False, opt=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        self.func = End2EndDP(gamma=opt.gamma, lbd_ord=opt.lbd_ord, lbd_dur=opt.lbd_dur, lbd_len=opt.lbd_len)
        self.sim = end2end_dp_v2
        
        self.max_violation = max_violation

    def forward(self, im, s,video_lengths, cap_lengths, mask=None):
        # compute image-sentence score matrix
        B = 0
        B = len(im)
        
        scores = self.sim(im, s, video_lengths, cap_lengths, self.func)
            
        diagonal = scores.diag().view(B, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

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


class PathLoss(nn.Module):
    def __init__(self,):
        super(PathLoss, self).__init__()
        self.loss_func = torch.nn.CrossEntropyLoss()
        
    def forward(self, im, s, paths, len_a, len_b):
        B = im.shape[0]
        M = im.shape[1]
        N = s.shape[1]

        loss = torch.zeros((B, )).to(im.device)
        for i in range(B):
            D = torch.zeros((1, M, N)).to(im.device)
            target = torch.zeros((1, M)).to(im.device).long()
            D[0, :, :] = (1 + F.normalize(im[i]).mm(F.normalize(s[i]).t())) / 2
            for j in range(len(paths[i])):
                target[0, j] = paths[i][j]
            loss[i] = self.loss_func(D[0, :len_a[i], :len_b[i]], target[0, :len_a[i]])

        return torch.mean(loss)
            
    

class COEMB_VCONV_TCONV_SDTW(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.opt = opt

        self.img_enc = EncoderImageSDTW_v2(opt.img_dim, opt.embed_size, opt.norm_method)
        self.txt_enc = EncoderImageSDTW_v2(opt.txt_dim, opt.embed_size, opt.norm_method)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation,
                                         opt=opt)
        
        self.criterion_path = PathLoss()
        
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
        # Set mini-batch dataset
        frames, gt_paths, video_lengths, vidoes_mask = videos
        frames = Variable(frames, requires_grad=False)
        if torch.cuda.is_available():
            frames = frames.cuda()

        vidoes_mask = Variable(vidoes_mask, requires_grad=False)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, gt_paths, video_lengths, vidoes_mask)
        
        # Set mini-batch dataset
        
        captions, _, cap_lengths, cap_masks = targets
        if captions is not None:
            captions = Variable(captions, requires_grad=False)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, requires_grad=False)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        text_data = (captions, _, cap_lengths, cap_masks)

        # Forward
        img_emb = self.img_enc(videos_data)
        cap_emb = self.txt_enc(text_data)
        
        
        
        ratio = img_emb.shape[1] * 1.0 / frames.shape[1]
        video_lengths = [int(l * ratio) for l in video_lengths]
        cap_lengths = [int(l * ratio) for l in cap_lengths]
        
        pooled_paths = []
        path_matrix = torch.zeros(())
        for bi in range(len(video_lengths)):
            v_len = video_lengths[bi]
            c_len = cap_lengths[bi]
            
            cur_path = []
            gt_path = gt_paths[bi]
            for vi in range(v_len):
                ci = int(gt_path[int(vi / ratio)]*ratio)
                if ci >= c_len:
                    break
                cur_path.append(ci)
            # print(v_len, c_len, cur_path)
            pooled_paths.append(cur_path)
                
        
        return img_emb, cap_emb, pooled_paths, video_lengths, cap_lengths

    def forward_loss(self, img_emb, cap_emb, paths, video_lengths, cap_lengths,mask=None, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss_dp = self.criterion(img_emb, cap_emb,video_lengths, cap_lengths, mask=mask)
        
        loss_path = 10  * self.criterion_path(img_emb, cap_emb, paths, video_lengths, cap_lengths)
        # print(loss_path)
        
        self.logger.update('Le_DP', loss_dp.data[0], len(img_emb))
        
        self.logger.update('Le_path', loss_path.data[0], len(img_emb))
        
        
        return loss_path  + loss_dp

    def train_emb(self, images, captions, ids, vids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, paths, video_lengths, cap_lengths = self.forward_emb(images, captions)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, paths, video_lengths, cap_lengths)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
