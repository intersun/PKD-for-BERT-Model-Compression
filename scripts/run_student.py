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
student_layer = sys.argv[2]
teacher_layer = sys.argv[3]

if teacher_layer == '12':
    bert_model = 'bert-base-uncased'
elif teacher_layer == '24':
    bert_model = 'bert-large-uncased'
else:
    raise ValueError(f'teacher_layer must be in [12, 24], now teacher_layer = {teacher_layer}')

run = 1 
all_lr = [5e-5, 2e-5, 1e-5]
all_T = [5, 10, 20]
all_alpha = [0.2, 0.5, 0.7]

all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

logging.info('will run {} for {} runs'.format(task, run))
logging.info('will run on %d GPUs' % n_gpu)

logging.info('all lr = {}'.format(all_lr))
logging.info('all T = {}'.format(all_T))
logging.info('all alpha = {}'.format(all_alpha))
tasks = task.split(',')

if 'race' in task:
    train_bs = 32 
    eval_bs = 16
    gradient_accumulate = 8 
    max_seq_length = 512
    do_eval = 'True'
else:
    train_bs = 32
    eval_bs = 32
    gradient_accumulate = 1
    max_seq_length = 128
    do_eval = 'False'

n_command = 0

for t in tasks:
    for _ in range(int(run)):
        for alpha, T, lr in itertools.product(all_alpha, all_T, all_lr):
            cmd = 'python %s/NLI_KD_training.py ' % PROJECT_FOLDER
            options = [
                '--learning_rate', str(lr),
                '--alpha', str(alpha),
                '--T', str(T)
            ]
            options += [
                '--task_name', t,
                '--bert_model', bert_model,
                '--train_batch_size', str(train_bs),
                '--eval_batch_size', str(eval_bs),
                '--gradient_accumulation_steps', str(gradient_accumulate),
                '--max_seq_length', str(max_seq_length),
                '--do_eval', str(do_eval),

                '--do_train', 'True',
                '--beta', '0.0',
                '--fp16', 'False',
                '--num_train_epochs', '4.0',
                '--kd_model', 'kd',
                '--log_every_step', '10',
                '--student_hidden_layers', str(student_layer),
            ]
            if alpha > 0:
                options += [
                    '--teacher_prediction', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{t}/{t}_normal_kd_teacher_{teacher_layer}layer_result_summary.pkl'),
                    '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{t}/kd_{student_layer}layer_teacher{teacher_layer}'),
                    ]
            else:
                options += [
                    '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{t}/ft_{student_layer}layer_teacher{teacher_layer}'),
                    ]

            cmd += ' '.join(options)
            all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d '%cur_gpu + cmd)
            cur_gpu += 1
            cur_gpu %= n_gpu

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)
