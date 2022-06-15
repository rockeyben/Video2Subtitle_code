import torch
import torch.utils.data as data
import numpy as np
import json as jsonmod
import h5py
import tqdm
import sys
import os
import re
import csv
import torchtext

VIDEO_MAX_LEN = 64

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

def make_2d_array(videos):
    video_lengths = [len(frame) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    
    return video_data

def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    videos, captions, idxs, video_ids = zip(*data)

    video_data = make_2d_array(videos)
    text_data = make_2d_array(captions)

    return video_data, text_data, idxs, video_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """
        
    def __init__(self, opt, phase='train'):

        self.vis_input_type = opt.vis_input_type
        self.data_path = opt.data_path 
        self.data_name = opt.data_name
        
        self.video_ids = []
        id_phase = 'gsm_train' if phase == 'train' else 'test'
        with open(os.path.join(self.data_path, self.data_name, '%s_ids.txt' % id_phase), 'r') as f:
            for line in f.readlines():
                self.video_ids.append(line.strip('\n'))

        print(phase, len(self.video_ids))
        
        self.length = len(self.video_ids)
        
        self.parse_script()
        self.init_glove()


    def init_glove(self):
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]

        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)

        if torch.__version__ == '0.3.1':
            word_embedding = torch.nn.Embedding(num_embeddings=400000, embedding_dim=300)
            word_embedding.weight = torch.nn.Parameter(vocab.vectors)
            word_embedding.weight.requires_grad = False
        else:  
            word_embedding = torch.nn.Embedding.from_pretrained(vocab.vectors)
        self.word_embedding = word_embedding
        self.vocab = vocab

    def parse_script(self):
        self.sen_dict = dict()
        self.sen_info = dict()
        
        with open(os.path.join(self.data_path, self.data_name, 'script.csv')) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                sen_id, v_id, s_time, e_time, sentence = row
                if float(s_time) >= float(e_time):
                    continue
                self.sen_dict[v_id] = []

        with open(os.path.join(self.data_path, self.data_name, 'script.csv')) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                sen_id, v_id, s_time, e_time, sentence = row
                v_id = str(v_id)
                sen_id = str(sen_id)
                if float(s_time) >= float(e_time):
                    continue
                self.sen_dict[v_id].append(sen_id)
                self.sen_info[sen_id] = [v_id, float(s_time)/1000.0, float(e_time)/1000.0, sentence]
        

    def __getitem__(self, index):
        
        video_id = self.video_ids[index]
            
        # video feature        
        obj_feat = np.load(os.path.join(self.data_path, self.data_name, 'feature_%s' % self.vis_input_type, video_id+'.npy'))
        frame_vec = []
        for ii in range(0, obj_feat.shape[0], 1):
            frame_vec.append(obj_feat[ii, :].astype(np.float32))
        video_data = torch.Tensor(frame_vec)

        # caption feature
        caption_vec = []
        text_feat_list = []
        for sid in self.sen_dict[video_id]:
            try:
                _, s_time, e_time, sentence = self.sen_info[sid]
                cop = re.compile("[^a-z^A-Z^0-9]")
                sentence = cop.sub(" ", sentence)
                sen_words = sentence.split()
                if len(sen_words) == 0:
                    continue
                word_idxs = torch.LongTensor([self.vocab.stoi.get(w.lower(), 400000) for w in sen_words])
                nlp_features = self.word_embedding(word_idxs).numpy()
                text_feat_list.append(nlp_features)
            except Exception as e:
                print(e)
                continue
    
        caption_vec = np.concatenate(text_feat_list, 0)
        caption = caption_vec.astype(np.float32)        
        caption = torch.Tensor(caption)
        return video_data, caption, index, video_id

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
        'test': Dataset4DualEncoding(opt, 'test'),
        'train': Dataset4DualEncoding(opt, 'train'),
             }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=opt.batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=False,
                                    collate_fn=collate_frame_gru_fn,
                                    num_workers=opt.workers)
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
        'test': Dataset4DualEncoding(opt, 'test'),
             }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=opt.batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    collate_fn=collate_frame_gru_fn,
                                    num_workers=1)
                        for x in ['test']}
    return data_loaders['test']

if __name__ == '__main__':
    pass
