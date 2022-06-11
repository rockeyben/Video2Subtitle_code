import pickle
import os
import time
import shutil
import torch
import data_movie_vrnn_tglove
from model import COEMB_VRNN_TRNN
from myevaluation import i2t, t2i, AverageMeter, LogCollector, encode_mydata

import logging
import tensorboard_logger as tb_logger

import argparse
import numpy as np

def read_dict(filepath):
    f = open(filepath,'r')  
    a = f.read()  
    dict_data = eval(a)  
    f.close()
    return dict_data

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
    # parser.add_argument('--num_layers', default=1, type=int,
    #                     help='Number of GRU layers.')
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
    
    opt = parser.parse_args()

    # opt.txt_dim = opt.word_dim
    # opt.visual_feat_dim = opt.img_dim
    
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

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

    train_loader, val_loader = data_movie_vrnn_tglove.get_data_loaders(opt)

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
            validate(opt, val_loader, model)
            exit()
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    no_impr_counter = 0
    lr_counter = 0
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best,filename='checkpoint_%d.pth.tar' % epoch, prefix=opt.logger_name + '/')

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)
        

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, video_idx = encode_mydata(
        model, val_loader, opt.log_step, logging.info)
    
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


def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
