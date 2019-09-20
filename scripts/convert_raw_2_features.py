# We convert the raw MRC data into torch tensors so we don't have to convert it everytime we need it

import pickle
import os

from src.race_data_processing import read_mrc_examples, convert_examples_to_features
from envs import HOME_DATA_FOLDER
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(os.path.join(HOME_DATA_FOLDER, 'models/pretrained/bert-base-uncased'),
                                          do_lower_case=True)

data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw')
data_output_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat')
# for task_name in ['race-high', 'race-merge', 'race-middle']:
for task_name in ['race-merge', 'race-high', 'race-middle']:
    print(task_name, end='...\t')
    output_folder = os.path.join(data_output_dir, task_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for s in ['train', 'dev', 'test']:
        train_examples = read_mrc_examples(os.path.join(data_dir, task_name, s + '.' + task_name + '.json'))
        features = convert_examples_to_features(train_examples, tokenizer, 512, True)

        with open(os.path.join(output_folder, s + '.' + task_name + '.pkl'), 'wb') as f:
            pickle.dump(features, f)
    print('Done')

