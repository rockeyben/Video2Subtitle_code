import os
import numpy as np
import torch.nn.functional as F
from end2end_dp_batch import compute_end2end_dp
import tqdm
import logging
import torch
from evaluation import i2t, t2i
import argparse

from sklearn import preprocessing
from ray.util.multiprocessing import Pool

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--test-path', default='/path/to/embs',
                    help='path to embs')
parser.add_argument('--save-path', default='/path/to/save',
                    help='save output scores')
parser.add_argument('--lbd_len', default=0.2, type=float,)
parser.add_argument('--lbd_dur', default=5, type=float,)
parser.add_argument('--lbd_ord', default=1, type=float,)
parser.add_argument('--gamma', default=0.1, type=float,)

opt = parser.parse_args()

test_path = opt.test_path 
save_path = opt.save_path

LBD_LEN = opt.lbd_len
LBD_DUR = opt.lbd_dur
LBD_ORD = opt.lbd_ord
GAMMA = opt.gamma

B = len(os.listdir(test_path)) // 2
# B = 100
num_workers = 32

p = Pool(num_workers)
args_list = []
for i in range(B):
    for j in range(B):
        args_list.append((os.path.join(test_path, '%d_v.npy' % i), 
                        os.path.join(test_path, '%d_a.npy' % j)))

def single_test(args):
    im_name, s_name = args
    
    im = np.load(im_name)
    s = np.load(s_name)

    D = 1 - F.normalize(torch.Tensor(im)).mm(F.normalize(torch.Tensor(s)).t())
    D = D.unsqueeze(0).numpy()
    
    window = compute_end2end_dp(D, np.array([im.shape[0]]), np.array([s.shape[0]]), 
        GAMMA, lbd_ord=LBD_ORD, lbd_dur=LBD_DUR, lbd_len=LBD_LEN)
    
    return window[0, -2, -2]

cost = list(tqdm.tqdm(p.imap(single_test, args_list), total=len(args_list)))

p.close()
p.join()

cost = np.array(cost).reshape((B, B)) / 100

r1, r5, r10, r100, medr, meanr= i2t(cost.T, n_captions=1)
logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1, r5, r10, medr, meanr))

np.save(save_path, cost.T)
