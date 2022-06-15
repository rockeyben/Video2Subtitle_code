import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import json
import re
import sys
import io
import pronouncing
import tqdm
import h5py
sys.path.append('/S4/MI/xueb/VST_feature_extracter')

from cca_dataset import MovieScript

corpus_dicts = ["cmudict-0.7b-simvecs", "speech2vec-50.vec"]
word_emb_file = corpus_dicts[0]

lookup = dict()

if word_emb_file == "cmudict-0.7b-simvecs":
    for i, line in enumerate(io.open(os.path.join('/S4/MI/xueb/VST_feature_extracter', word_emb_file), encoding="latin1")):
        line = line.strip()
        word, vec_s = line.split("  ")
        vec = [float(n) for n in vec_s.split()]
        lookup[word] = vec

def pooling(ipt, length):
    # (N_FEAT, T)
    layer = torch.nn.AdaptiveAvgPool1d(length)
    ipt = layer(ipt.unsqueeze(0)).squeeze(0)
    return ipt.data.numpy()



def get_embedding(id_list, DS, opt):
    sen_info = DS.sen_info
    sen_dict = DS.sen_dict
    err_word = set()
    WORD_DUR = opt.word_dur
    T = opt.T
    AUD = []
    TXT = []
    pbar = tqdm.tqdm(total=len(id_list))
    total_face_len = 0
    blank_face_len = 0
    for vid in id_list:
        pbar.update(1)

        # collect audio/visual feature, ignore blank postions
        if opt.vis_input_type in ['face-lip', 'face-emo']:
            aud_feat = DS.get_face_embedding(vid)
            if len(aud_feat) == 0:
                continue
            aud_feat_tensor = torch.FloatTensor(aud_feat)
            cur_T = aud_feat_tensor.size(1)
        else:
            aud_feat = audio_file[vid][:]
            aud_feat_tensor = torch.FloatTensor(aud_feat)
            if opt.vis_input_type in ['lip_visible', 'action', 'object', 'scene']:
                aud_feat_tensor = aud_feat_tensor.transpose(1,0)
            cur_T = aud_feat_tensor.size(1)
        aud_feat_list = []

        # generate text feature
        text_feat_list = []
        sen_ids = sen_dict[vid]
        
        for sid in sen_ids:
            _, s_time, e_time, duration, sentence = sen_info[sid]
            
            s_idx = int(s_time/duration*cur_T)
            e_idx = min(int(e_time/duration*cur_T), cur_T)
            if e_idx - s_idx <= 1:
                continue
            aud_feat_list.append(aud_feat_tensor[:, s_idx:e_idx])

            strs = None
            if opt.nlp_type == 'word':
                cop = re.compile("[^a-z^A-Z^0-9]")
                if word_emb_file == "cmudict-0.7b-simvecs":
                    cop = re.compile("[^'^a-z^A-Z^0-9]")
                sentence = cop.sub(" ", sentence)
                strs = sentence.split()
                if len(strs) == 0:
                    continue
                word_embds = []
                for w in strs:
                    try:
                        if WORD_DUR:
                            phone_list = pronouncing.phones_for_word(w)
                            phone = phone_list[0].split()
                            word_embds.extend([lookup[w.upper()]] * len(phone))
                        else:
                            word_embds.append(lookup[w.upper()])
                    except:
                        err_word.add(w.upper())
                        continue
                if len(word_embds) == 0:
                    aud_feat_list = aud_feat_list[:-1]
                    continue
                sen_feat = torch.FloatTensor(np.array(word_embds)).transpose(1,0)
                sen_feat = pooling(sen_feat, e_idx-s_idx)
                text_feat_list.append(torch.FloatTensor(sen_feat))
            elif opt.nlp_type == 'letter':
                cop = re.compile("[^ ^a-z^A-Z^0-9]")
                sentence = cop.sub("", sentence)
                strs = [c for c in sentence]
                if len(strs) == 0:
                    continue
                onehot = str2onehot(strs)
                onehot = torch.FloatTensor(onehot)
                onehot = pooling(onehot, e_idx-s_idx)
                text_feat_list.append(torch.FloatTensor(onehot))
            elif opt.nlp_type == 'semantic':
                sen_words = sentence.split()
                if len(sen_words) == 0:
                    continue
                word_idxs = torch.LongTensor([ds.vocab.stoi.get(w.lower(), 400000) for w in sen_words])
                nlp_features = ds.word_embedding(word_idxs).transpose(1, 0)
                sen_feat = pooling(nlp_features, e_idx-s_idx)
                text_feat_list.append(torch.FloatTensor(sen_feat))
            elif opt.nlp_type == 'emotion':
                nlp_features = text_emo_file[sid][:]
                nlp_features = torch.FloatTensor(nlp_features).transpose(1,0)
                sen_feat = pooling(nlp_features, e_idx - s_idx)
                text_feat_list.append(torch.FloatTensor(sen_feat))
        if len(text_feat_list) == 0:
            continue
        aud_feat = torch.cat(aud_feat_list, 1)
        text_feat = torch.cat(text_feat_list, 1)
        
        aud_feat = pooling(aud_feat, T)
        text_feat = pooling(text_feat, T)
        
        AUD.append(aud_feat)
        TXT.append(text_feat)

    AUD = np.concatenate(AUD, 1).transpose(1,0)
    TXT = np.concatenate(TXT, 1).transpose(1,0)

    return AUD, TXT


def get_embedding_global(id_list, DS, opt):
    AUD = []
    TXT = []
    pbar = tqdm.tqdm(total=len(id_list))
    
    audio_file = h5py.File(os.path.join(DS.data_dir, DS.hdf5_file), 'r')
    bert_sen_file = h5py.File(os.path.join(DS.data_dir, 'bert_sen.hdf5'), 'r')
    bert_sen_id_file = h5py.File(os.path.join(DS.data_dir, 'bert_sen_id.hdf5'), 'r')
    
    cap2vid = []
    vid2cap = dict()
    sen_dict = DS.sen_dict
    sen_info = DS.sen_info
    
    for vid in id_list:
        pbar.update(1)
        # collect audio/visual feature, ignore blank postions
        if opt.vis_input_type == 'face-emo':
            aud_feat = DS.get_face_embedding_global(vid, fusion_type='avg')
            if len(aud_feat) == 0:
                continue
        else:
            aud_feat = audio_file[vid][:]

        #aud_feat = np.mean(aud_feat, axis=0, keepdims=True)
        aud_feat = aud_feat[0:1, :]
        sen_ids = sen_dict[vid]
        # generate text feature
        if opt.nlp_type == 'semantic':
            text_feat_list = []
            for sid in sen_ids:
                _, s_time, e_time, duration, sentence = sen_info[sid]
                cop = re.compile("[^a-z^A-Z^0-9]")
                sentence = cop.sub(" ", sentence)
                sen_words = sentence.split()
                if len(sen_words) == 0:
                    continue
                word_idxs = torch.LongTensor([ds.vocab.stoi.get(w.lower(), 400000) for w in sen_words])
                nlp_features = ds.word_embedding(word_idxs).numpy()
                text_feat_list.append(nlp_features)
            if len(text_feat_list) == 0:
                continue
            text_feat = np.concatenate(text_feat_list, 0)
            text_feat = np.mean(text_feat, axis=0, keepdims=True)
        elif opt.nlp_type == 'bert':
            text_feat = bert_sen_file[vid][:]
        elif opt.nlp_type == 'bert_id':
            text_feat_list = []
            AUD.append(aud_feat)
            tmp_vid_index = len(AUD) - 1
            tmp_project = []
            for sid in sen_ids:
                try:
                    text_feat = bert_sen_id_file[sid][:]
                except Exception as e:
                    print(e)
                    continue
                TXT.append(text_feat)
                tmp_sid_index = len(TXT) - 1
                cap2vid.append(tmp_vid_index)
                tmp_project.append(tmp_sid_index)
            vid2cap[tmp_vid_index] = tmp_project
            continue
        elif opt.nlp_type == 'emotion':
            text_feat_list = []
            for sid in sen_ids:
                _, s_time, e_time, duration, sentence = sen_info[sid]
                nlp_features = text_emo_file[sid][:]
                nlp_features = torch.FloatTensor(nlp_features)
                text_feat_list.append(nlp_features)
            text_feat = torch.cat(text_feat_list, 0).transpose(1, 0)
            text_feat = pooling(text_feat, T).transpose(1, 0)
        
        AUD.append(aud_feat)
        TXT.append(text_feat)

    AUD = np.concatenate(AUD, 0)
    TXT = np.concatenate(TXT, 0)
    print(AUD.shape, TXT.shape)

    return AUD, TXT, cap2vid, vid2cap



class PrecompDataset(data.Dataset):
    def __init__(self, vis, txt):
        self.vis = vis
        self.txt = txt
            
    def __getitem__(self, index):
        image = torch.Tensor(self.vis[index])
        target = torch.Tensor(self.txt[index])
        return image, target, index, index

    def __len__(self):
        return self.vis.shape[0]
    
class MultiSentencePrecompDataset(data.Dataset):
    def __init__(self, vis, txt, cap2vid, vid2cap):
        self.vis = vis
        self.txt = txt
        self.cap2vid = cap2vid
        self.vid2cap = vid2cap
            
    def __getitem__(self, index):
        sid = index
        vid = self.cap2vid[index]
        image = torch.Tensor(self.vis[vid])
        target = torch.Tensor(self.txt[sid])
        return image, target, index, vid

    def __len__(self):
        return self.txt.shape[0]


def get_loaders(opt):
    ds = MovieScript(opt, 'train')
    ds_test = MovieScript(opt, 'test')
    train_ids = ds.annotations
    test_ids = ds_test.annotations
    train_vis, train_txt, train_cap2vid, train_vid2cap = get_embedding_global(train_ids, ds, opt)
    print(train_vis.shape, train_txt.shape)
    print(len(train_cap2vid))
    test_vis, test_txt, test_cap2vid, test_vid2cap = get_embedding_global(test_ids, ds_test, opt)
    print(len(test_cap2vid))
    
    if opt.nlp_type == 'bert_id':
        train_set = MultiSentencePrecompDataset(train_vis, train_txt, train_cap2vid, train_vid2cap)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                pin_memory=True)
        
        test_set = MultiSentencePrecompDataset(test_vis, test_txt, test_cap2vid, test_vid2cap)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                pin_memory=True)       
        
    else:
        train_set = PrecompDataset(train_vis, train_txt)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                pin_memory=True)
        
        test_set = PrecompDataset(test_vis, test_txt)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                pin_memory=True)
    
    return train_loader, test_loader


def get_test_loaders(opt):
    ds_test = MovieScript(opt, 'test')
    test_ids = ds_test.annotations
    test_vis, test_txt, test_cap2vid, test_vid2cap = get_embedding_global(test_ids, ds_test, opt)
    print(len(test_cap2vid))
    
    if opt.nlp_type == 'bert_id':
        
        test_set = MultiSentencePrecompDataset(test_vis, test_txt, test_cap2vid, test_vid2cap)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                pin_memory=True)       
        
    else:
        
        test_set = PrecompDataset(test_vis, test_txt)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                pin_memory=True)
    
    return test_loader
