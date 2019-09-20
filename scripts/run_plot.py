import os
import sys
import pandas as pd
import collections
import torch
from multiprocessing import Pool

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_FOLDER)

from envs import HOME_DATA_FOLDER
from src.utils import run_process
from src.nli_data_processing import parse_filename


bert_model = 'bert-base-uncased'
task = sys.argv[1]
log_every_step = sys.argv[2]

log_file = {
    'teacher': os.path.join(PROJECT_FOLDER, 'result/glue/result_summary/teacher_12layer_mean.csv'),
    'kg': os.path.join(PROJECT_FOLDER, 'result/glue/result_summary/kd_6layer_mean.csv'),
    'pkd': os.path.join(PROJECT_FOLDER, 'result/glue/result_summary/pkd_6layer_mean.csv')
}

all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0

res = []
for name in log_file:
    df = pd.read_csv(log_file[name])
    for i, v in df.iterrows():
        model, task_check, nlayer, lr, T, alpha, beta, bs, run = parse_filename(v[0])
        if task == task_check:
            if beta > 0:
                tmp = model.split('.')
                model, normalize = tmp[0] + '.' + tmp[1], tmp[2]
            res.append(v[0])

            cmd = 'python %s/NLI_KD_training_GeneratePlot.py ' % PROJECT_FOLDER
            options = [
                '--log_every_step', log_every_step,
                '--learning_rate', str(lr),
                '--alpha', str(alpha),
                '--T', str(T),
                '--beta', str(beta),
                '--student_hidden_layers', str(nlayer),
                '--kd_model', model,
            ]
            options += [
                '--task_name', task,
                '--bert_model', bert_model,
                '--train_batch_size', str(bs),
                '--eval_batch_size', '32',
                '--gradient_accumulation_steps', '1',
                '--max_seq_length', '128',
                '--do_train', 'True',
                '--fp16', 'True',
                '--num_train_epochs', '4.0',
                '--output_dir', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{task}/plot'),
            ]
            if alpha > 0:
                if beta > 0:
                    options += [
                        '--teacher_prediction', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{task}/{task}_patient_kd_teacher_12layer_result_summary.pkl'),
                        '--normalize_patience', normalize,
                        '--fc_layer_idx', '1,3,5,7,9',
                    ]
                else:
                    options += [
                        '--teacher_prediction', os.path.join(HOME_DATA_FOLDER, f'outputs/KD/{task}/{task}_normal_kd_teacher_12layer_result_summary.pkl'),
                    ]

            cmd += ' '.join(options)
            all_cmds[cur_gpu].append('CUDA_VISIBLE_DEVICES=%d '%cur_gpu + cmd)
            cur_gpu += 1
            cur_gpu %= n_gpu

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

pool = Pool(processes=n_gpu)
pool.map(run_process, run_cmd)
