#encoding=utf-8

from __future__ import division
from __future__ import absolute_import
import argparse
import time

import glob
import os
# import sys
# sys.path.insert(0, os.path.dirname(__file__))
# sys.path.insert(-1, os.path.dirname(__file__))
import random
import signal

from BertSum.src.prepro import customized


import torch
from pytorch_pretrained_bert import BertConfig

import distributed

from BertSum.src import models
from BertSum.src.models import data_loader, model_builder
from BertSum.src.models.data_loader import load_dataset
from BertSum.src.models.model_builder import Summarizer
from BertSum.src.models.trainer import build_trainer
from BertSum.src.others.logging import logger, init_logger

def data_create(oracle_mode = 'greedy', raw_path = '../../json_data/', save_path = '../../bert_data/', text = '', log_file='../../logs/preprocess.log', dataset='test', n_cpus=2):
    customized.format_to_lines(raw_path, text)
    customized.format_to_bert(dataset, raw_path, save_path, n_cpus)



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def test(args, device_id, pt, step):
    model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    # logger.info('Loading checkpoint from %s' % test_from)
    
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    # print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    # model.load_state_dict(torch.load(test_from))
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter,step)


def result(args):
    with open(args.result_path+"_step1000.txt","r") as f:
        file=f.read()
    return [x.strip() for x in file.split("<q>")]



def do_BertSum():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bert_data_path", default='../bert_data/ndm_sample')
    parser.add_argument("-model_path", default='../models/bert_transformer')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-bert_config_path", default='../bert_config_uncased_base.json')
    parser.add_argument("-batch_size", default=1000, type=int)
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-save_checkpoint_steps", default=200, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/bert_transformer')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='../models/bert_transformer/model_step_1000.pt')
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.chdir(os.path.dirname(__file__))

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    cp = args.test_from
    try:
        step = int(cp.split('.')[-2].split('_')[-1])
    except:
        step = 0
    test(args, device_id, cp, step)
    return result(args)


if __name__ == '__main__':
    data_create(text='코로나19 신규 환자 수가 64명 추가돼 이달 들어 나흘 연속 두 자릿수를 기록했습니다. 그러나 지역사회에서의 집단 감염이 계속되고 있어 방역 당국은 여전히 긴정하고 있습니다. 이덕영 기자입니다. 오늘 0시 기준 코로나19 신규 확진 환자는 64명입니다. 국내발생 47명 해외 유입 17명입니다. 이달 들어 나흘 연속 하루 신규 환자수가 두 자릿수를 나타내고 있는 겁니다. 하지만 추석 연휴를 받아 진단검사 수가 줄어든 가운데 전국적인 인구 이동으로 조용한 전파가 이뤄스 가능성도 있어 방역당국은 여전히 긴장을 늦추지 못하고 있습니다. 지역별로는 서울과 인천 경기 등 수도권에서 38명 부산 5명 경북 4명 대구 대전 충북 100명 등이 양성 판정을 받았습니다. 종교시설과 요양원 의료기관 등 사양한 장소에서의 집단발생이 계속되고 있는 상황입니다. 어제 정오 기준 서울 도봉구 단화병원 관련 확진자는 46명으로 늘어났고 경기도 포천의 소망공동체 유항원 관련해서도 14명이 확진됐습니다. 또 인천 미추홀구 소망교의 교인 10명과 부산 연재구 건강용품 4명의 관련 24명이 확진 판정을 받았습니다. 추석 귀성의 영향으로 부산을 방문한 서울과 울산 거주자 2명이 확진됐습니다. 한편 위중중증 환자는 105명이고 사망자는 1명이 추가돼 모두 421명으로 늘어났습니다. 엠비씨뉴스 이덕영입니다.')
    summary = do_BertSum()
    print(summary)