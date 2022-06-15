import os
import time
import shutil
import torch
import logging
import argparse
import numpy as np

import data_movie_vrnn_tglove
from model import  COEMB_VRNN_TRNN
from myevaluation import i2t, t2i, AverageMeter, LogCollector, encode_mydata

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/path/to/data',
                        help='path to datasets')
    parser.add_argument('--data-name', default='youtube',
                        help='(youtube|movie)')
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
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--n_caption', type=int, default=1, help='number of captions of each image/video (default: 1)')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate (default: 0.2)')

    # add v2, video rnn
    parser.add_argument('--vis_input_type', default='object', help='cue')
    parser.add_argument('--visual_feat_dim', default=4096, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument('--visual_rnn_size', type=int, default=1024, help='visual rnn encoder size')
    parser.add_argument('--visual_kernel_num', default=512, type=int, help='number of each kind of visual kernel')
    parser.add_argument('--visual_kernel_sizes', default='2-3-4-5', type=str, help='dash-separated kernel size to use for visual convolution')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')

    # add v3, text rnn
    parser.add_argument('--word_dim', type=int, default=300, help='word embedding dimension')
    parser.add_argument('--text_rnn_size', type=int, default=512, help='text rnn encoder size. (default: 1024)')
    parser.add_argument('--text_kernel_num', default=512, type=int, help='number of each kind of text kernel')
    parser.add_argument('--text_kernel_sizes', default='2-3-4', type=str, help='dash-separated kernel size to use for text convolution')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-2048', help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-2048', help='visual fully connected layers  for common space learning. (default: 0-2048)')
    parser.add_argument('--concate', type=str, default='full', help='feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    

    parser.add_argument('--save-path', default='/path/to/save',
                        help='save output scores')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(os.path.dirname(opt.save_path))
    except Exception as e:
        pass

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
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
            exit()
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, video_idx = encode_mydata(
        model, val_loader, opt.log_step, logging.info)
    
    cost = - np.dot(cap_embs, img_embs.T)
    cost = cost.astype(np.float16)

    r1, r5, r10, r100, medr, meanr= i2t(cost, n_captions=opt.n_caption)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))

    currscore = r1 + r5 + r10

    np.save(opt.save_path, cost)

    return currscore



if __name__ == '__main__':
    main()
