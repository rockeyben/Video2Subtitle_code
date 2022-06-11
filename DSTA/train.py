import pickle
import os
from posixpath import realpath
import time
import shutil
import torch
import dataset
from model import COEMB_VCONV_TCONV_SDTW
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_mydata
import logging
import tensorboard_logger as tb_logger

from end2end_dp_batch import End2EndDP, compute_end2end_dp
import torch.nn.functional as F
from tqdm import tqdm
import ray

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int, help='Dimensionality of the image embedding.')
    parser.add_argument('--txt_dim', type=int, default=300, help='word embedding dimension')
    parser.add_argument('--n_caption', type=int, default=1, help='number of captions of each image/video (default: 1)')
    
    parser.add_argument('--phase', default='train',
                        help='(train|test)')
    parser.add_argument('--norm-method', default='bn',
                        help='(bn|none)')
    parser.add_argument('--test-num', default=-1, type=int,
                        help='set test number')
    parser.add_argument('--parallel-test', action='store_true',
                        help='use multiprocessing to test whole dataset')
    parser.add_argument('--dump-name', default='tmp')
    
    
    parser.add_argument('--lbd_len', default=0.2, type=float,)
    parser.add_argument('--lbd_dur', default=5, type=float,)
    parser.add_argument('--lbd_ord', default=1, type=float,)
    parser.add_argument('--gamma', default=0.1, type=float,)
    

    opt = parser.parse_args()

    try:
        os.makedirs(opt.dump_name)
    except Exception as e:
        print(e)
        pass

    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    try:
        os.makedirs(os.path.join(opt.logger_name, 'figures'))
    except Exception as e:
        print(e)
        pass

    # Construct the model
    model = COEMB_VCONV_TCONV_SDTW(opt)

    if opt.phase == 'test' or opt.phase == 'val':
        # Evaluate the Model
        
        val_loader = dataset.get_test_loaders(opt)
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
            
            
    elif opt.phase == 'train':

        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))

        # Train the Model
        train_loader, val_loader = dataset.get_data_loaders(opt)
        
        rsum = validate(opt, val_loader, model)
        
        best_rsum = 0
        no_impr_counter = 0
        lr_counter = 0
        for epoch in range(opt.num_epochs):
            adjust_learning_rate(opt, model.optimizer, epoch)

            # train for one epoch
            train(opt, train_loader, model, epoch)

            rsum = validate(opt, val_loader, model)
            # rsum = 0

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


def train(opt, train_loader, model, epoch):
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



def test_end2end_dp(im, s, real_paths, img_lens, cap_lens, opt):
    
    criterion = End2EndDP(gamma=opt.gamma, lbd_ord=opt.lbd_ord, lbd_dur=opt.lbd_dur, lbd_len=opt.lbd_len)
    B = im.shape[0]
    M = im.shape[1]
    N = s.shape[1]

    score = torch.zeros((B, B))

    for i in tqdm(range(B)):
        D = torch.zeros((B, M, N))
        cur_len_a = torch.zeros((B))
        cur_len_b = torch.zeros((B))
        for j in range(B):
            cur_len_a[j] = img_lens[i]
            cur_len_b[j] = cap_lens[j]
            D[j, :, :] = 1 - F.normalize(im[i]).mm(F.normalize(s[j]).t())
            if i == j:
                dist_mat = D[j, :int(img_lens[i]), :int(cap_lens[j])]
                
                plt.figure()
                ax_gram = plt.axes()
                ax_gram.imshow(dist_mat, origin='lower')
                ax_gram.axis('off')
                ax_gram.autoscale(False)
                
                xpath = []
                ypath = []
                for xx in range(len(real_paths[i])):
                    xpath.append(xx)
                    ypath.append(real_paths[i][xx])
                ax_gram.plot(ypath, xpath, "r-", linewidth=1.)
                plt.savefig(os.path.join(opt.logger_name, 'figures', '%d.png' % i))

        score[:, i] = criterion(D, cur_len_a.int(), cur_len_b.int())
        score[:, i] = score[:, i] / 100

    return score.numpy()


def parallel_test(im, s, real_paths, img_lens, cap_lens, test_path=None, num_workers=32):
    
    B = im.shape[0]
    M = im.shape[1]
    N = s.shape[1]
    C = im.shape[2]
    
    for i in range(B):
        np.save(os.path.join(test_path, '%d_v.npy' % i), im[i][:int(img_lens[i])])            
        np.save(os.path.join(test_path, '%d_a.npy' % i), s[i][:int(cap_lens[i])])                
    
    score = np.zeros((B, B))
    
    return score
    
def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, real_paths, img_lens, cap_lens, video_idx = encode_mydata(
        model, val_loader, opt.log_step, logging.info)
    
    if opt.parallel_test:
        cost = parallel_test(img_embs, cap_embs, real_paths, img_lens, cap_lens, test_path=opt.dump_name)
    else:
        cost = test_end2end_dp(torch.Tensor(img_embs), torch.Tensor(cap_embs), real_paths, img_lens, cap_lens, opt)
        
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
