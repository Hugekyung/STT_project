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
import json
import math
import random
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

import Levenshtein as Lev 

import label_loader
from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler

from models import EncoderRNN, DecoderRNN, Seq2Seq


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
            except:
                error_lst.append(i)
                pass
        print(error_lst)

    aver_loss = total_loss / total_num
    aver_cer = float(total_dist / total_length) * 100
    return aver_loss, aver_cer, transcripts_list


def main():
    os.chdir(os.path.dirname(__file__))
    os.chdir('../')

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    model_name = 'LAS'
    # Dataset
    test_file_list = ['data/Youtube_test/youtube_test.json']
    labels_path = 'data/kor_syllable.json'
    dataset_path = 'data/Youtube_test'

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


    print(">> Test dataset : ", test_file_list)
    testLoader_dict = {}


    for test_file in test_file_list:
        testData_list = []
        with open(test_file, 'r', encoding='utf-8') as f:
            testData_list = json.load(f)
        
        test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                          dataset_path=dataset_path, 
                                          data_list=testData_list,
                                          char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                          normalize=True)
        testLoader_dict[test_file] = AudioDataLoader(test_dataset, batch_size=1, num_workers=num_workers)


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

    for test_file in test_file_list:
        test_loader = testLoader_dict[test_file]
        test_loss, test_cer, transcripts_list = evaluate(model, test_loader, criterion, device, save_output=True)

        for line in transcripts_list:
            print('STT : ' + line.split('\t')[0])
            print('정답 : ' + line.split('\t')[1])
            print('-'*100)

        print("Test {} CER : {}".format(test_file, test_cer))

if __name__ == "__main__":
    main()
