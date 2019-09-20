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


# task = 'CoLA,SST-2,MRPC,QQP,MNLI,QNLI,RTE'
task = sys.argv[1]
bert_model = sys.argv[2]   #'bert-base-uncased'
assert bert_model in ['bert-base-uncased', 'bert-large-uncased'], 'bert models needs to be bert-base-uncased or bert-large-uncased'
n_layer = 12 if 'base' in bert_model else 24

run = 1
all_lr = [1e-6, 5e-5, 2e-5]
all_lr = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

logging.info('will run {} for {} runs'.format(task, run))
logging.info('will run on %d GPUs' % n_gpu)

logging.info('all lr = {}'.format(all_lr))

if 'race' in task:
    pass
else:
    tasks = task.split(',')
    for t in tasks:
        for _ in range(int(run)):
            for lr in all_lr:
                cmd = 'python %s/NLI_KD_training.py ' % PROJECT_FOLDER
                options = ['--learning_rate', str(lr)]
                options += [
                    '--task_name', t,
                    '--alpha', '0.0',
                    '--T', '10.0',
                    '--bert_model', bert_model,
                    '--train_batch_size', '32',
                    '--eval_batch_size', '32',
                    '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{t}/teacher_{n_layer}layer'),
                    '--do_train', 'True',
                    '--beta', '0.0',
                    '--max_seq_length', '128',
                    '--fp16', 'True',
                    '--num_train_epochs', '4.0',
                    '--kd_model', 'kd',
                    '--log_every_step', '10',
                ]

                cmd += ' '.join(options)
                all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d ' % cur_gpu + cmd)
                cur_gpu += 1
                cur_gpu %= n_gpu

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

# print(run_cmd)
pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)
