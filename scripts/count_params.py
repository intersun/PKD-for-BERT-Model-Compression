import logging
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from BERT.pytorch_pretrained_bert.modeling import BertConfig
from src.modeling import BertForMultipleChoiceEncoder, FCClassifierMultipleChoice
from src.utils import count_parameters

from envs import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = argparse.Namespace(bert_model='bert-base-uncased',
                          output_all_encoded_layers=False
                          )


BERT_PRETRAINED_FOLDER = os.path.join(HOME_DATA_FOLDER, 'models/pretrained', args.bert_model)
assert os.path.isdir(BERT_PRETRAINED_FOLDER), 'BERT_PRETRAINED_FOLDER init fail: {}'.format(BERT_PRETRAINED_FOLDER)

config = BertConfig(os.path.join(BERT_PRETRAINED_FOLDER, 'bert_config.json'))

bert_encoder = dict()
for l in range(1, 13):
    bert_encoder[l] = BertForMultipleChoiceEncoder(config, output_all_encoded_layers=args.output_all_encoded_layers,
                                                   num_hidden_layers=l)

params_count = list()
for l in range(1, 13):
    n_param_encoder = count_parameters(bert_encoder[l].bert.encoder)
    n_param_pooler = count_parameters(bert_encoder[l].bert.pooler)
    n_param_embedding = count_parameters(bert_encoder[l].bert.embeddings)
    n_params_total = count_parameters(bert_encoder[l])
    n_params_total_debug = n_param_encoder + n_param_pooler + n_param_embedding
    assert n_params_total == n_params_total_debug, 'total num params error'

    params_count.append([l, n_param_embedding, n_param_pooler, n_param_encoder, n_params_total])
params_count_df = pd.DataFrame(params_count, columns=['n_layers', '#embedding', '#pooler', '#encoder', '#total'])
print('num params per encoder = %d' % count_parameters(bert_encoder[1].bert.encoder))
print(params_count_df)
