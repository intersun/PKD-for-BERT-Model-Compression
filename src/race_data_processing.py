import logging
import json
import string
import os
import torch
import pickle
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from src.modeling import BertForMultipleChoiceEncoder, FCClassifierMultipleChoice
from envs import HOME_DATA_FOLDER


logger = logging.getLogger(__name__)


class MRCExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 mrc_id,
                 passage,
                 question,
                 answers,
                 label=None):
        self.mrc_id = mrc_id
        self.passage = passage
        self.question = question
        self.answers = answers
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"mrc_id: {self.mrc_id}",
            f"passage: {self.passage}",
            f"question: {self.question}",
            f"answer_0: {self.answers[0]}",
            f"answer_1: {self.answers[1]}",
            f"answer_2: {self.answers[2]}",
            f"answer_3: {self.answers[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label
    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_mrc_examples(input_file):

    with open(input_file, 'r') as f:
        data = json.load(f)

    examples = []
    answer_mapping = {string.ascii_uppercase[i]: i for i in range(4)}

    for (i, passage_id) in enumerate(data.keys()):
        for question_id in range(len(data[passage_id]['questions'])):
            guid = '%s-%s' % (passage_id, str(question_id))
            article = data[passage_id]['article']
            question = data[passage_id]['questions'][question_id]
            choices = data[passage_id]['options'][question_id]
            label = answer_mapping[data[passage_id]['answers'][question_id].upper()]
            examples.append(
                MRCExample(mrc_id=guid, passage=article, question=question, answers=choices, label=label))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.passage)
        start_ending_tokens = tokenizer.tokenize(example.question)

        choices_features = []
        for ending_index, ending in enumerate(example.answers):

            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 0:
            logger.info("*** Example ***")
            logger.info(f"mrc_id: {example.mrc_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id=example.mrc_id,
                choices_features=choices_features,
                label=label
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', task_name)
    feat_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat', task_name)
    examples = read_mrc_examples(os.path.join(raw_data_dir, set_name.lower() + '.' + task_name + '.json'))
    if batch_size is None:
        batch_size = args.train_batch_size if set_name.lower() == 'train' else args.eval_batch_size

    features = pickle.load(open(os.path.join(feat_data_dir, set_name.lower() + '.' + task_name + '.pkl'), 'rb'))

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)

    if knowledge is not None:
        all_knowledge = torch.tensor(knowledge, dtype=torch.float)
        if extra_knowledge is None:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_knowledge)
        else:
            layer_index = [int(i) for i in args.fc_layer_idx.split(',')]
            extra_knowledge_tensor = torch.stack([torch.FloatTensor(extra_knowledge[int(i)]) for i in layer_index]).transpose(0, 1)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_knowledge, extra_knowledge_tensor)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)
    return examples, dataloader, all_label


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def init_race_model(task_name, output_all_layers, num_hidden_layers, config):
    assert 'race' in task_name, f'task name needs to be a race task, instead of {task_name}'
    encoder_bert = BertForMultipleChoiceEncoder(config, output_all_encoded_layers=output_all_layers,
                                                        num_hidden_layers=num_hidden_layers)
    classifier = FCClassifierMultipleChoice(config, 4, config.hidden_size, 0)
    return encoder_bert, classifier
