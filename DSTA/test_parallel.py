import os
import numpy as np
import torch.nn.functional as F
from end2end_dp_batch import compute_end2end_dp
import tqdm
import logging
import torch
from evaluation import i2t, t2i
from sklearn import preprocessing
from ray.util.multiprocessing import Pool

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


exp_name = 'youtube_end2end'
# exp_name = 'youtube_pool4x_dp'
# exp_name = 'movie_end2end'
# exp_name = 'movie_end2end_lambda0.5'
# exp_name = 'movie_end2end_lambda0.1'

if exp_name == 'youtube_end2end':
    test_path = 'runs/test_emb'
    save_path = 'score/youtube_end2end_dp_score.npy'
    lbd = 0.2
elif exp_name == 'youtube_pool4x_dp':
    test_path = 'tmp/youtube_pooled_emb' 
    save_path = 'score/youtube_pool4x_dp_score.npy'
    lbd = 0.2
elif exp_name == 'movie_end2end':
    test_path = 'runs/movie_v2_train_pooling_pathloss_dploss_resume/test_embs'
    save_path = 'score/movie_end2end_dp_score_v2.npy'
    lbd = 0.2
elif exp_name == 'movie_end2end_lambda0.5':
    test_path = 'runs/movie_v2_train_pooling_pathloss_dploss_lambda0.5_resume/test_embs_v2'
    save_path = 'score/movie_end2end_dp_lambda0.5_score.npy'
    lbd = 0.5
elif exp_name == 'movie_end2end_lambda0.1':
    test_path = 'runs/movie_v2_train_pooling_pathloss_dploss_lambda0.1_floatfix_lr1e-3/test_embs'
    save_path = 'score/movie_end2end_dp_lambda0.1_score.npy'
    lbd = 0.1

# B = len(os.listdir(test_path)) // 2
B = 100
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
    
    # print(D)
    
    # im = sklearn.preprocessing.normalize(im)
    # s = sklearn.preprocessing.normalize(s)
    # D = 1 - np.dot(im, s.T)

    # print(D.shape, np.array([im.shape[0]]), np.array([s.shape[0]]))
    
    window = compute_end2end_dp(D, np.array([im.shape[0]]), np.array([s.shape[0]]), 0.1, lbd=lbd)
    # print(window.shape)
    # print(window)
    
    return window[0, -2, -2]

cost = list(tqdm.tqdm(p.imap(single_test, args_list), total=len(args_list)))
# cost = single_test((os.path.join(test_path, '%d_v.npy' % 0), 
#                         os.path.join(test_path, '%d_a.npy' % 0)))
# print(cost)

p.close()
p.join()

cost = np.array(cost).reshape((B, B)) / 100

r1, r5, r10, r100, medr, meanr= i2t(cost.T, n_captions=1)
logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1, r5, r10, medr, meanr))
# image retrieval
r1i, r5i, r10i, r100i, medri, meanri = t2i(
    cost.T, n_captions=1)
logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1i, r5i, r10i, medri, meanr))
# sum of recalls to be used for early stopping
currscore = r1 + r5 + r10 + r1i + r5i + r10i

# np.save(save_path, cost)