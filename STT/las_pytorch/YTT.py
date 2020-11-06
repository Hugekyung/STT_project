"""
Copyright 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import glob
import json
import math
import random
import shutil
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

import Levenshtein as Lev 

import label_loader
from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler

from models import EncoderRNN, DecoderRNN, Seq2Seq
from chunks_loader import Youtube_to_Wav


char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 


def get_distance(ref_labels, hyp_labels):
    total_dist = 0
    total_length = 0
    transcripts = []
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])

        transcripts.append('{hyp}\t{ref}'.format(hyp=hyp, ref=ref))

        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 

    return total_dist, total_length, transcripts

def evaluate(model, data_loader, criterion, device, save_output=False):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    transcripts_list = []

    model.eval()
    error_lst = []
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader)):
            try:
                feats, scripts, feat_lengths, script_lengths = data

                feats = feats.to(device)
                scripts = scripts.to(device)
                feat_lengths = feat_lengths.to(device)

                src_len = scripts.size(1)
                target = scripts[:, 1:]

                logit = model(feats, feat_lengths, None, teacher_forcing_ratio=0.0)
                logit = torch.stack(logit, dim=1).to(device)
                y_hat = logit.max(-1)[1]

                logit = logit[:,:target.size(1),:] # cut over length to calculate loss
                loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
                total_loss += loss.item()
                total_num += sum(feat_lengths).item()

                dist, length, transcripts = get_distance(target, y_hat)
                cer = float(dist / length) * 100

                total_dist += dist
                total_length += length
                if save_output == True:
                    transcripts_list += transcripts
                total_sent_num += target.size(0)
            except Exception as e:
                error_lst.append(i)
                pass
        print(error_lst)

    return transcripts_list


def youtube_to_wav(link):
    ytw = Youtube_to_Wav()
    ytw.link_to_chunks(link)


def wav_to_text():
    os.chdir(os.path.dirname(__file__))
    os.chdir('../')

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    model_name = 'LAS'
    # Dataset
    target_file_lst = glob.glob('data/youtube/*/chunks/*.pcm')
    target_dic = []
    for chunk in target_file_lst:
        target_dic.append({'wav':chunk, 'text': '1'})
    with open('data/youtube/target_file.json', 'w', encoding='utf-8') as json_file:
        json.dump(target_dic, json_file)
    target_file_list = ['data/youtube/target_file.json']
    labels_path = 'data/kor_syllable.json'
    dataset_path = ''

    # Hyperparameters
    rnn_type = 'lstm'
    encoder_layers = 3
    encoder_size = 512
    decoder_layers = 2
    decoder_size = 512
    dropout = 0.3
    bidirectional = True
    num_workers = 4
    max_len = 80

    # Audio Config
    sample_rate = 16000
    window_size = .02
    window_stride = .01

    # System
    save_folder = 'models'
    model_path = 'models/AIHub_train/LSTM_512x3_512x2_AIHub_train/final.pth'
    cuda = True
    seed = 123456

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    char2index, index2char = label_loader.load_label_json(labels_path)
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    device = torch.device('cuda' if cuda else 'cpu')

    audio_conf = dict(sample_rate=sample_rate,
                      window_size=window_size,
                      window_stride=window_stride)


    print(">> Target dataset : ", target_file_list)
    targetLoader_dict = {}


    for target_file in target_file_list:
        targetData_list = []
        with open(target_file, 'r', encoding='utf-8') as f:
            targetData_list = json.load(f)
        
        target_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                          dataset_path=dataset_path, 
                                          data_list=targetData_list,
                                          char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                          normalize=True)
        targetLoader_dict[target_file] = AudioDataLoader(target_dataset, batch_size=1, num_workers=num_workers)

    input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
    enc = EncoderRNN(input_size, encoder_size, n_layers=encoder_layers,
                     dropout_p=dropout, bidirectional=bidirectional, 
                     rnn_cell=rnn_type, variable_lengths=False)

    dec = DecoderRNN(len(char2index), max_len, decoder_size, encoder_size,
                     SOS_token, EOS_token,
                     n_layers=decoder_layers, rnn_cell=rnn_type, 
                     dropout_p=dropout, bidirectional_encoder=bidirectional)


    model = Seq2Seq(enc, dec)
    os.makedirs(save_folder, exist_ok=True)


    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    print("Loading checkpoint model %s" % model_path)
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    print('Model loaded')

    model = model.to(device)

    print(model)
    print("Number of parameters: %d" % Seq2Seq.get_param_size(model))

    result = []
    for target_file in target_file_list:
        target_loader = targetLoader_dict[target_file]
        transcripts_list = evaluate(model, target_loader, criterion, device, save_output=True)
        result.append(' '.join([line.split('\t')[0] for line in transcripts_list]))
    
    lst = os.listdir('data/youtube')
    for x in lst:
        if x != 'target_file.json':
            shutil.rmtree('data/youtube/'+x)

    return result

if __name__ == "__main__":
    test_lst = ['3IBycan2RS8', 'BIykJPi8VJo', 'JFfpruMuOVc', 'JIchmo9vz9k', 'Oq2uajg5Ovw', 'qxodlEHVeBg', 'W1pCrrvESIY', 'ysUGUH95iMw', 'n8S_qusdbvE', 'Nkr8LQS1Odw']
    test_result = []
    for test in test_lst:
        youtube_to_wav('https://youtu.be/{}'.format(test))
        result = wav_to_text()
        print(result)
        test_result.append(result)
    print(test_result)
