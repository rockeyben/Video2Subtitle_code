from __future__ import print_function
import os
import pickle

import time
import numpy as np
from collections import OrderedDict
import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_dtwdata(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    video_index = None
    cap_aligns = None
    for i, (images, captions, ids, vids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = ['']*len(data_loader.dataset)
            cap_embs = ['']*len(data_loader.dataset)
            video_index = ['']*len(data_loader.dataset)
            cap_aligns = ['']*len(data_loader.dataset)

        # preserve the embeddings by copying from gpu and converting to numpy
        for j, idx in enumerate(ids):
            img_embs[idx] = img_emb[j].detach().cpu().numpy().copy()
            cap_embs[idx] = cap_emb[j].detach().cpu().numpy().copy()
            video_index[idx] = vids[j]
            cap_aligns[idx] = data_loader.dataset.cap_alignments[vids[j]]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        del images, captions
    
    return img_embs, cap_embs, cap_aligns, video_index


def encode_mydata(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    img_lens = None
    cap_lens = None
    video_index = None
    real_paths = []
    pbar = tqdm.tqdm(total=len(data_loader))
    for i, (images, captions, ids, vids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        pbar.update()

        # compute the embeddings
        img_emb, cap_emb, paths, img_len, cap_len = model.forward_emb(images, captions,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1), cap_emb.size(2)))
            img_lens = np.zeros((len(data_loader.dataset),))
            cap_lens = np.zeros((len(data_loader.dataset),))
            video_index = ['']*len(data_loader.dataset)

        # preserve the embeddings by copying from gpu and converting to numpy
        for j, idx in enumerate(ids):
            #print(img_emb.shape, cap_emb.shape)
            img_embs[idx] = img_emb[j].detach().cpu().numpy().copy()
            cap_embs[idx] = cap_emb[j].detach().cpu().numpy().copy()
            img_lens[idx] = img_len[j]
            cap_lens[idx] = cap_len[j]
            video_index[idx] = vids[j]
            real_paths.append(paths[j])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        del images, captions
    
    #print(img_embs.shape)
    # print(img_lens, cap_lens)
    return img_embs, cap_embs, real_paths, img_lens, cap_lens, video_index



def t2i(c2i, n_captions=1):
    ranks = np.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == i / n_captions)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, r100, medr, meanr])

def i2t(c2i, n_captions=1):
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds/n_captions == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, r100, medr, meanr])



def i2t_mini(c2i, index):
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds == index[i])[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, r100, medr, meanr])


def i2t_one(c2i, index):
    
    inds = np.argsort(c2i)
    rank = np.where(inds == index)[0][0]

    return rank

def print_rank(ranks):
    ranks = np.array(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    print(r1, r5, r10, r100, medr, meanr)
    
    return map(float, [r1, r5, r10, r100, medr, meanr])