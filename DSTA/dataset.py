import torch
import torch.utils.data as data
import numpy as np
import tqdm
import os
import re

import argparse


MAX_LEN = 1024

def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    if vid_id.endswith('.jpg') or vid_id.endswith('.mp4'):
        vid_id = vid_id[:-4]
    return vid_id


def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def make_2d_array(videos, paths=None):
    video_lengths = [len(frame) for frame in videos]
    frame_vec_len = len(videos[0][0])

    vidoes = torch.zeros(len(videos), MAX_LEN, frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    
    gt_paths = torch.zeros(len(videos), MAX_LEN)

    for i, frames in enumerate(videos):
            end = min(video_lengths[i], MAX_LEN)
            video_lengths[i] = end
            vidoes[i, :end, :] = frames[:end,:]
            vidoes_mask[i,:end] = 1.0
            
            if paths:
                path = paths[i]
                path = path / 640.0 / 2
                for vp in range(end):                    
                    if vp * 2 >= len(path):
                        break
                    gt_paths[i][vp] = path[vp*2]
                        
    video_data = (vidoes, gt_paths, video_lengths, vidoes_mask)
    
    return video_data

def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    videos, captions, paths, idxs, video_ids = zip(*data)

    video_data = make_2d_array(videos, paths)
    text_data = make_2d_array(captions)

    return video_data, text_data, idxs, video_ids


class Dataset4End2EndDP(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, opt, phase='train'):

        data_path = opt.data_path
        data_name = opt.data_name
        
        self.v_feat_dir = os.path.join(data_path, data_name, 'feature_lip')
        self.a_feat_dir = os.path.join(data_path, data_name, 'feature_tts')
        self.path_dir = os.path.join(data_path, data_name, 'gt_path')
        
        
        self.video_ids = []
        id_phase = 'dsta_train' if phase == 'train' else 'test'
        with open(os.path.join(data_path, data_name, '%s_ids.txt' % id_phase), 'r') as f:
            for line in f.readlines():
                self.video_ids.append(line.strip('\n'))

        if phase == 'val':
            self.video_ids = self.video_ids[:100]
        
        self.length = len(self.video_ids)
        
        print(phase, self.length)
        
        
    def __getitem__(self, index):
        video_id = self.video_ids[index]
            
        # video feature        
        v_feat = np.load(os.path.join(self.v_feat_dir, video_id+'.npy'))
        video_data = torch.Tensor(v_feat[0]).permute((1, 0))

        a_feat = np.load(os.path.join(self.a_feat_dir, video_id+'.npy'))
        caption = torch.Tensor(a_feat[0]).permute((1, 0))
        
        path = np.load(os.path.join(self.path_dir, video_id+'.npy'))
        
        return video_data, caption, path, index, video_id

    def __len__(self):
        return self.length
     

def get_data_loaders(opt):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {
        'test': Dataset4End2EndDP(opt, 'val'),
        'train': Dataset4End2EndDP(opt, 'train'),
             }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=opt.batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=False,
                                    collate_fn=collate_frame_gru_fn,
                                    num_workers=4)
                        for x in ['train', 'test']}
    return data_loaders['train'], data_loaders['test']

def get_test_loaders(opt):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {
        'test': Dataset4End2EndDP(opt, opt.phase),
             }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=opt.batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    collate_fn=collate_frame_gru_fn,
                                    num_workers=1)
                        for x in ['test']}
    return data_loaders['test']
