import numpy as np
import argparse
import os
import datetime
import json
import h5py
import csv
import cv2
from shutil import copyfile
import torch
import torch.nn.functional as F
from DSTA.evaluation import i2t_one, print_rank
from DSTA.end2end_dp_batch import compute_end2end_dp_path

def load_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        success, frame = capture.read()
        if frame is None:
            break
        frames.append(frame)
    return frames


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='/path/to/data',
                    help='path to datasets')
parser.add_argument('--data-name', default='youtube',
                    help='(youtube|movie)')
parser.add_argument('--dsta', default='/path/to/score.npy',
                    help='dsta alignment score')
parser.add_argument('--dsta-embs', default='/path/to/test_embs',
                    help='dsta embs')
parser.add_argument('--object', default='/path/to/score.npy',
                    help='object score')
parser.add_argument('--scene', default='/path/to/score.npy',
                    help='scene score')
parser.add_argument('--action', default='/path/to/score.npy',
                    help='action score')
parser.add_argument('--output-path', default='demo_results')
parser.add_argument('--test-id', default='0', 
                    help='select a vid in /path/to/test_ids.txt')
parser.add_argument('--w_d', default=10.0,
                    help='dsta weight')
parser.add_argument('--w_o', default=1.0,
                    help='object weight')
parser.add_argument('--w_s', default=1.0,
                    help='scene weight')
parser.add_argument('--w_a', default=1.0,
                    help='action weight')
parser.add_argument('--K', default=3, type=int,
                    help='top K fake ids')
parser.add_argument('--lbd_len', default=0.2, type=float,)
parser.add_argument('--lbd_dur', default=5, type=float,)
parser.add_argument('--lbd_ord', default=1, type=float,)
parser.add_argument('--gamma', default=0.1, type=float,)

opt = parser.parse_args()

try:
    os.makedirs(opt.output_path)
except:
    pass

Mat_D = np.load(opt.dsta)
Mat_O = np.load(opt.object)
Mat_S = np.load(opt.scene)
Mat_A = np.load(opt.action)


# prepare data
with open(os.path.join(opt.data_path, opt.data_name, 'video_info.json'), 'r') as f:
    video_info = json.load(f)

with open(os.path.join(opt.data_path, opt.data_name, 'script_noblank.json'), 'r') as f:
    clean_words_file = json.load(f)

id_list = []
with open(os.path.join(opt.data_path, opt.data_name, 'test_ids.txt'), 'r') as f:
    for line in f.readlines():
        id_list.append(line.strip())

sen_dict = dict()
sen_info = dict()

with open(os.path.join(opt.data_path, opt.data_name, 'script.csv')) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        sen_id, v_id, s_time, e_time, sentence = row
        if float(s_time) >= float(e_time):
            continue
        sen_dict[v_id] = []

with open(os.path.join(opt.data_path, opt.data_name, 'script.csv')) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        sen_id, v_id, s_time, e_time, sentence = row
        v_id = str(v_id)
        sen_id = str(sen_id)
        if float(s_time) >= float(e_time):
            continue
        sen_dict[v_id].append(sen_id)
        sen_info[sen_id] = [v_id, float(s_time)/1000.0, float(e_time)/1000.0, sentence]

assert opt.test_id in id_list

# get total fusion results
ranks = []
for i in range(len(id_list)):
    score = Mat_D[:, i] * opt.w_d + Mat_O[:, i] * opt.w_o + Mat_S[:, i] * opt.w_s + Mat_A[:, i] * opt.w_a
    
    r = i2t_one(score, i)
    ranks.append(r)

print('total fusion result')
print_rank(ranks)

# get fake retreival 
tid = id_list.index(opt.test_id)
score = Mat_D[:, tid] * opt.w_d + Mat_O[:, tid] * opt.w_o + Mat_S[:, tid] * opt.w_s + Mat_A[:, tid] * opt.w_a
topk = np.argsort(score)[:opt.K]

# prepare video
spk_score_file = h5py.File(os.path.join(opt.data_path, opt.data_name, 'speaking_score.hdf5'), 'r')
video_name = video_info[opt.test_id][0]
FPS = video_info[opt.test_id][1]
raw_frames = load_video(os.path.join(opt.data_path, opt.data_name, 'raw_video', video_name + '.mp4'))
spk_score = spk_score_file[opt.test_id][:]
noblank_frames = []

sen_ids = sen_dict[opt.test_id]
T = spk_score.shape[0]
for sid in sen_ids:
    _, s_time, e_time, sentence = sen_info[sid]
    duration = video_info[opt.test_id][2]
    s_idx = int(s_time / duration * T)
    e_idx = int(min(e_time, duration) / duration * T)

    for fi in range(s_idx, e_idx):
        if spk_score[fi] > 0:
            noblank_frames.append(raw_frames[fi])

videoWriter = cv2.VideoWriter(os.path.join(opt.output_path, '%s.avi' % opt.test_id) , cv2.VideoWriter_fourcc(*'XVID') , FPS, (640, 360))
for frame in noblank_frames:
    videoWriter.write(frame)

duration = len(noblank_frames) / float(FPS)

# prepare re-generated subtitle
for ri, fake_id in enumerate(topk):

    f_vid = id_list[fake_id] 

    sens = clean_words_file[f_vid]
    fa = np.load(os.path.join(opt.dsta_embs, '%d_a.npy' % fake_id))
    fv = np.load(os.path.join(opt.dsta_embs, '%d_v.npy' % fake_id))

    D = 1 - F.normalize(torch.Tensor(fv)).mm(F.normalize(torch.Tensor(fa)).t())
    D = D.unsqueeze(0).numpy()

    window, path = compute_end2end_dp_path(D, np.array([fv.shape[0]]), np.array([fa.shape[0]]), 
        opt.gamma, lbd_ord=opt.lbd_ord, lbd_dur=opt.lbd_dur, lbd_len=opt.lbd_len)
    
    rate = float(fa.shape[0]) / sens[-1][1]
    len_v = fv.shape[0]
    
    srt_file = open(os.path.join(opt.output_path, 'vid%s_top%d_fid%s.srt' % (opt.test_id, ri, f_vid)), 'w')
    
    no = 1
    for info in sens:
        s_ts, e_ts, sen = info
        s_ts = int(s_ts *rate)
        e_ts = int(e_ts *rate)
        
        s_vs = np.where(path <= s_ts)[0][-1]
        e_vs = np.where(path <= e_ts)[0][-1]
        
        s_sub = float(s_vs) / float(len_v) * duration
        e_sub = float(e_vs) / float(len_v) * duration
        
        zero = datetime.datetime.strptime('00:00:00', '%H:%M:%S')
        sub_st = zero + datetime.timedelta(seconds=s_sub)
        sub_ed = zero + datetime.timedelta(seconds=e_sub)
        str_st = datetime.datetime.strftime(sub_st, '%H:%M:%S')
        str_ed = datetime.datetime.strftime(sub_ed, '%H:%M:%S')

        srt_file.write(str(no) + '\n')
        srt_file.write(str_st + ',433' + ' --> ' + str_ed + ',433\n')
        srt_file.write(sen+ '\n\n')
        no += 1