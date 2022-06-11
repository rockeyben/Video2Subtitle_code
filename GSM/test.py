import pickle
import os
import time
import shutil
import torch
import data_movie_vrnn_tglove
from model import VSE, COEMB_VRNN_TRNN
from myevaluation import i2t, t2i, AverageMeter, LogCollector, encode_mydata

import logging
import tensorboard_logger as tb_logger

import h5py
import argparse
import numpy as np
from bigfile import BigFile

def read_dict(filepath):
    f = open(filepath,'r')  
    a = f.read()  
    dict_data = eval(a)  
    f.close()
    return dict_data

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=2048, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--nlp_type', default='bert',
                        help='path to datasets')
    parser.add_argument('--vis_input_type', default='scene',
                        help='path to datasets')
    parser.add_argument('--n_caption', type=int, default=1, help='number of captions of each image/video (default: 1)')
    
    
    # add v2, video rnn
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate (default: 0.2)')
    parser.add_argument('--visual_rnn_size', type=int, default=1024, help='visual rnn encoder size')
    parser.add_argument('--visual_kernel_num', default=512, type=int, help='number of each kind of visual kernel')
    parser.add_argument('--visual_kernel_sizes', default='2-3-4-5', type=str, help='dash-separated kernel size to use for visual convolution')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-2048', help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-2048', help='visual fully connected layers  for common space learning. (default: 0-2048)')
    parser.add_argument('--concate', type=str, default='full', help='feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    
    # add v3, text rnn

    parser.add_argument('--word_dim', type=int, default=300, help='word embedding dimension')
    parser.add_argument('--text_rnn_size', type=int, default=512, help='text rnn encoder size. (default: 1024)')
    parser.add_argument('--text_kernel_num', default=512, type=int, help='number of each kind of text kernel')
    parser.add_argument('--text_kernel_sizes', default='2-3-4', type=str, help='dash-separated kernel size to use for text convolution')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    
    opt = parser.parse_args()
    opt.data_dir = '/S4/MI/xueb/VST_feature_extracter/dataset/youtube_clean'
    opt.mini_data = -1
    opt.word_dur = True
    opt.init_glove = True
    opt.txt_dim = opt.word_dim
    print(opt)
    
    print(opt.vis_input_type)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    opt.visual_feat_dim = opt.img_dim
    
    
    # mapping layer structure
    opt.text_kernel_sizes = map(int, opt.text_kernel_sizes.split('-'))
    opt.visual_kernel_sizes = map(int, opt.visual_kernel_sizes.split('-'))
    
    opt.text_mapping_layers = map(int, opt.text_mapping_layers.split('-'))
    opt.visual_mapping_layers = map(int, opt.visual_mapping_layers.split('-'))
    if opt.concate == 'full':
        opt.text_mapping_layers[0] = opt.word_dim + opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes) 
        opt.visual_mapping_layers[0] = opt.visual_feat_dim + opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
    elif opt.concate == 'reduced':
        opt.text_mapping_layers[0] = opt.text_rnn_size*2 + opt.text_kernel_num * len(opt.text_kernel_sizes) 
        opt.visual_mapping_layers[0] = opt.visual_rnn_size*2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
    else:
        raise NotImplementedError('Model %s not implemented'%opt.model)


    # Load data loaders
    #train_loader, val_loader = data_movie_vrnn_tglove.get_data_loaders(opt)
    val_loader = data_movie_vrnn_tglove.get_test_loaders(opt)
    # Construct the model
    model = COEMB_VRNN_TRNN(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            rsum = validate(opt, val_loader, model)
            print(rsum)
            exit()
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


def validate(opt, val_loader, model):
    video_info = val_loader.dataset.video_info
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, video_idx = encode_mydata(
        model, val_loader, opt.log_step, logging.info)
    
    feat_v_file = h5py.File(os.path.join(os.path.dirname(opt.resume), 'feat_v.hdf5'), 'w')    
    feat_t_file = h5py.File(os.path.join(os.path.dirname(opt.resume), 'feat_t.hdf5'), 'w')    
    
    print(img_embs.shape)
    
    for _, vid in enumerate(video_idx):
        vname = video_info[vid][0]
        feat_v_file[vname] = img_embs[_]
        feat_t_file[vname] = cap_embs[_]
    
    cost = - np.dot(cap_embs, img_embs.T)

    # caption retrieval
    r1, r5, r10, r100, medr, meanr= i2t(cost, n_captions=opt.n_caption)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    r1i, r5i, r10i, r100i, medri, meanri = t2i(
        cost, n_captions=opt.n_caption)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
