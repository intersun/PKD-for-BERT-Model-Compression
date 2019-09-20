import os
import sys
import collections
import torch
import logging
import itertools
from multiprocessing import Pool

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_FOLDER)

from envs import HOME_DATA_FOLDER, PREDICTION_FOLDER
from src.utils import run_process


task = 'MRPC,RTE,SST-2,QNLI,MNLI,QQP'
bert_model = 'bert-large-uncased'
assert bert_model in ['bert-base-uncased', 'bert-large-uncased'], 'bert models needs to be bert-base-uncased or bert-large-uncased'
n_layer = 12 if 'base' in bert_model else 24


all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

logging.info('will run on %d GPUs' % n_gpu)



tasks = task.split(',')
for t in tasks:

    cmd = f'python {PROJECT_FOLDER}/scripts/run_glue_benchmark.py {t} teacher:train,dev True '
    cmd += os.path.join(PROJECT_FOLDER, 'result/glue/result_summary/teacher_24layer_all.csv')
    cmd += ' ' + bert_model
    all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d ' % cur_gpu + cmd)
    cur_gpu += 1
    cur_gpu %= n_gpu

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

# print(run_cmd)
pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)
