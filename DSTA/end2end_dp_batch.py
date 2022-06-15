from ntpath import realpath
import numpy as np
import torch
from torch.autograd import Variable, Function
from numba import jit
import time


@jit(nopython = True, cache=False)
def compute_end2end_dp(D, len_a, len_b, gamma, lbd_ord=1, lbd_dur=5, lbd_len=0.2):
    B = D.shape[0]
    l1 = D.shape[1]
    l2 = D.shape[2]
    
    sigma = 1.0
    margin = 1.0
    ws = max(5, np.abs(l1 - l2) + 1)
    
    window = np.ones((B, l1+2, l2+2)) * np.inf
    length_dist = np.zeros((B, l1+2, l2+2))
    order_penalty = np.zeros((B, l2+2, l2+2))
    
    for k in range(B):
        M = len_a[k]
        N = len_b[k]
        for i in range(M+2):
            for j in range(N+2):
                err = np.abs(i - j)
                length_dist[k, i, j] = 1 - np.exp(-err**2/(2*(sigma**2)*(j+1)))

        length_dist[k, :, :] = length_dist[k, :, :] / float(M) * lbd_dur + np.abs(M - N) / float(M) * lbd_len

    
    for k in range(B):
        N = len_b[k]
        for i in range(N+2): # curr step
            for j in range(N+2): # prev step
                order_penalty[k, i, j] = lbd_ord * max(margin - (i - j), 0)
    
    window[:, 0, 0] = 0
    
    
    for k in range(B):
        M = len_a[k]
        N = len_b[k]
        ws = max(5, np.abs(M - N) + 1)
        for i in range(1, M + 2):

            curr_lb = max(1, i - ws)
            curr_rb = min(N+2, i + ws)
            prev_lb = max(0, curr_lb - 1)
            prev_rb = min(N+2, curr_rb - 1)
            
            for j_curr in range(curr_lb, curr_rb):
                    
                r0 = -(window[k, i-1, prev_lb:prev_rb] + \
                    order_penalty[k, j_curr, prev_lb:prev_rb]) / gamma
                
                rmax = np.max(r0)
                rsum = np.sum(np.exp(r0 - rmax))
                softmin = - gamma * (np.log(rsum) + rmax)
                window[k, i, j_curr] = D[k, i - 1, j_curr - 1] + \
                    length_dist[k, i, j_curr] + \
                    softmin

    return window

@jit(nopython = True, cache=True)
def compute_end2end_dp_backward(D_, R, len_a, len_b, gamma, lbd_ord=1, lbd_dur=5, lbd_len=0.2):
    B = D_.shape[0]
    l1 = D_.shape[1]
    l2 = D_.shape[2]
    
    sigma = 1
    margin = 1.0
    ws = max(5, np.abs(l1 - l2) + 1)
    
    D = np.zeros((B, l1 + 2, l2 + 2))
    E = np.zeros((B, l1 + 2, l2 + 2))
    D[:, 1:l1 + 1, 1:l2 + 1] = D_
    for k in range(B):
        E[k, len_a[k], len_b[k]] = 1

    for k in range(B):
        R[k, len_a[k]+1, :] = -np.inf
    for k in range(B):
        R[k, len_a[k]+1, len_b[k]+1] = R[k, len_a[k], len_b[k]]

    length_dist = np.zeros((B, l1+2, l2+2))
    order_penalty = np.zeros((B, l2+2, l2+2))

    for k in range(B):
        M = len_a[k]
        N = len_b[k]
        for i in range(M+2):
            for j in range(N+2):
                err = np.abs(i - j)
                length_dist[k, i, j] = 1 - np.exp(-err**2/(2*(sigma**2)*(j+1)))

        length_dist[k, :, :] = length_dist[k, :, :] / float(M) * lbd_dur + np.abs(M - N) / float(M) * lbd_len

    
    for k in range(B):
        N = len_b[k]
        for i in range(N+2): # curr step
            for j in range(N+2): # prev step
                order_penalty[k, i, j] = lbd_ord * max(margin - (i - j), 0)
        

    for k in range(B):
        M = len_a[k]
        N = len_b[k]
        ws = max(5, np.abs(M - N) + 1)
        
        for i in range(M-1, 0, -1):
            
            curr_lb = max(0, i - ws)
            curr_rb = min(N+1, i + ws)
            post_lb = max(1, curr_lb + 1)
            post_rb = min(N+1, curr_rb + 1)
            
            for j_curr in range(curr_rb, curr_lb-1, -1):
                
                r0 = (R[k, i+1, post_lb:post_rb] - \
                    order_penalty[k, post_lb:post_rb, j_curr] - \
                    R[k, i, j_curr] - length_dist[k, i+1, post_lb:post_rb] - \
                    D[k, i+1, post_lb:post_rb]) / gamma
                r = np.exp(r0)
                
                E[k, i, j_curr] = np.sum(E[k, i+1, post_lb:post_rb] * r)

    
    return E[:, 1:l1+1, 1:l2+1]

class _End2EndDP(Function):
    @staticmethod
    def forward(ctx, D, len_a, len_b, gamma, lbd_ord=1, lbd_dur=5, lbd_len=0.2):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)
        
        lbd_ord = torch.Tensor([lbd_ord]).to(dev).type(dtype)
        lbd_dur = torch.Tensor([lbd_dur]).to(dev).type(dtype)
        lbd_len = torch.Tensor([lbd_len]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        len_a_ = len_a.detach().cpu().numpy()
        len_b_ = len_b.detach().cpu().numpy()
        g_ = gamma.item()
        lbd_ord_ = lbd_ord.item()
        lbd_dur_ = lbd_dur.item()
        lbd_len_ = lbd_len.item()
        
        R = torch.Tensor(compute_end2end_dp(D_, len_a_, len_b_, g_, lbd_ord_, lbd_dur_, lbd_len_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, len_a, len_b, gamma, lbd_ord, lbd_dur, lbd_len)

        B = len_a.shape[0]        
        ret = torch.zeros((B)).to(dev).type(dtype)
        for k in range(B):
            ret[k] = R[k, len_a_[k], len_b_[k]]

        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, len_a, len_b, gamma, lbd_ord, lbd_dur, lbd_len = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        len_a_ = len_a.detach().cpu().numpy()
        len_b_ = len_b.detach().cpu().numpy()
        g_ = gamma.item()
        lbd_ord_ = lbd_ord.item()
        lbd_dur_ = lbd_dur.item()
        lbd_len_ = lbd_len.item()
        
        E = torch.Tensor(compute_end2end_dp_backward(D_, R_,  len_a_, len_b_, g_, lbd_ord_, lbd_dur_, lbd_len_)).to(dev).type(dtype)
        
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None, None, None, None


class End2EndDP(torch.nn.Module):
    def __init__(self, gamma=1.0, lbd_ord=1, lbd_dur=5, lbd_len=0.2):
        super(End2EndDP, self).__init__()
        self.gamma = gamma
        self.lbd_ord = lbd_ord
        self.lbd_dur = lbd_dur
        self.lbd_len = lbd_len
        self.func_dp = _End2EndDP.apply

    def forward(self, D, len_a, len_b):
        out_score = self.func_dp(D, len_a, len_b, self.gamma, self.lbd_ord, self.lbd_dur, self.lbd_len)
        return out_score



@jit(nopython = True, cache=True)
def compute_end2end_dp_path(D, len_a, len_b, gamma, lbd_ord=1, lbd_dur=5, lbd_len=0.2):
    B = D.shape[0]
    l1 = D.shape[1]
    l2 = D.shape[2]
    
    sigma = 1
    margin = 1.0
    ws = max(5, np.abs(l1 - l2) + 1)
    
    window = np.ones((B, l1+2, l2+2)) * np.inf
    length_dist = np.zeros((B, l1+2, l2+2))
    order_penalty = np.zeros((B, l2+2, l2+2))
    
    for k in range(B):
        M = len_a[k]
        N = len_b[k]
        for i in range(M+2):
            for j in range(N+2):
                err = np.abs(i - j)
                length_dist[k, i, j] = 1 - np.exp(-err**2/(2*(sigma**2)*(j+1)))
                
        length_dist[k, :, :] = length_dist[k, :, :] / float(M) * lbd_dur + np.abs(M - N) / float(M) * lbd_len
                
    
    for k in range(B):
        N = len_b[k]
        for i in range(N+2): # curr step
            for j in range(N+2): # prev step
                order_penalty[k, i, j] = lbd_ord * max(margin - (i - j), 0)
    
    window[:, 0, 0] = 0
    
    
    path = np.zeros((B, l1+2, l2+2), dtype=np.int32)
    
    
    for k in range(B):

        M = len_a[k]
        N = len_b[k]
        ws = max(5, np.abs(M - N) + 1)
        for i in range(1, M + 2):

            curr_lb = max(1, i - ws)
            curr_rb = min(N+2, i + ws)
            prev_lb = max(0, curr_lb - 1)
            prev_rb = min(N+2, curr_rb - 1)
            
            for j_curr in range(curr_lb, curr_rb):
                    
                r0 = -(window[k, i-1, prev_lb:prev_rb] + \
                    order_penalty[k, j_curr, prev_lb:prev_rb]) / gamma

                path[k, i, j_curr] = np.argmax(r0) + prev_lb
                rmax = np.max(r0)
                rsum = np.sum(np.exp(r0 - rmax))
                softmin = - gamma * (np.log(rsum) + rmax)
                window[k, i, j_curr] = D[k, i - 1, j_curr - 1] + \
                    length_dist[k, i, j_curr] + \
                    softmin

    real_path = np.zeros((l1+1,))
    p = l1
    q = l2
    real_path[l1] = l2
    while p > 1:
        q = path[0, p, q]
        p = p - 1
        real_path[p] = q
    
    return window, real_path
