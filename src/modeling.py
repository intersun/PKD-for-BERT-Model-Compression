import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from BERT.pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertEmbeddings, BertEncoder

logger = logging.getLogger(__name__)


class BertModelNoPooler(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelNoPooler, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        return encoded_layers


class BertForMultipleChoiceEncoder(BertPreTrainedModel):
    # Dropout layer is in classifier/decoder

    def __init__(self, config, output_all_encoded_layers=False, num_hidden_layers=None):
        super(BertForMultipleChoiceEncoder, self).__init__(config)
        self.config = config
        if num_hidden_layers is not None:
            logger.info('num hidden layer is set as %d' % num_hidden_layers)
            config.num_hidden_layers = num_hidden_layers

        logger.info("Model config {}".format(config))
        self.bert = BertModel(config)
        self.output_all_encoded_layers = output_all_encoded_layers
        self.mode = 'finetune'
        self.apply(self.init_bert_weights)

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'freeze':
            for params in self.bert.parameters():
                params.requires_grad = False
        elif mode == 'finetune':
            pass
        elif mode == 'pooler':
            # freeze all layers first, then unfreeze pooler layer
            for params in self.bert.parameters():
                params.requires_grad = False
            for params in self.bert.pooler.parameters():
                params.requires_grad = True
        else:
            raise ValueError('mode in BertEncoder should be either finetune, freeze or pooler')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if self.output_all_encoded_layers:
            full_output, pool_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask,
                                                 output_all_encoded_layers=True)

            # only return [CLS] token for now
            return [full_output[i][:, 0] for i in range(len(full_output))], pool_output
        else:
            _, pool_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask,
                                       output_all_encoded_layers=False)
            return None, pool_output


class FCClassifierMultipleChoice(BertPreTrainedModel):
    def __init__(self, config, num_choices, hidden_size=100, n_layers=0):
        super(FCClassifierMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, 'fc%d' % i, nn.Linear(hidden_size, hidden_size))

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, encoded_feat):
        encoded_feat = self.dropout(encoded_feat)
        for i in range(self.n_layers):
            encoded_feat = getattr(self, 'fc%d' % i)(encoded_feat)
        logits = self.classifier(encoded_feat)
        reshaped_logits = logits.view(-1, self.num_choices)
        return reshaped_logits


class BertForSequenceClassificationEncoder(BertPreTrainedModel):
    def __init__(self, config, output_all_encoded_layers=False, num_hidden_layers=None, fix_pooler=False):
        super(BertForSequenceClassificationEncoder, self).__init__(config)
        if num_hidden_layers is not None:
            logger.info('num hidden layer is set as %d' % num_hidden_layers)
            config.num_hidden_layers = num_hidden_layers

        logger.info("Model config {}".format(config))
        if fix_pooler:
            self.bert = BertModelNoPooler(config)
        else:
            self.bert = BertModel(config)
        self.output_all_encoded_layers = output_all_encoded_layers
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        if self.output_all_encoded_layers:
            full_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
            return [full_output[i][:, 0] for i in range(len(full_output))], pooled_output
        else:
            _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            return None, pooled_output


class FCClassifierForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, hidden_size, n_layers=0):
        super(FCClassifierForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, 'fc%d' % i, nn.Linear(hidden_size, hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, encoded_feat):
        encoded_feat = self.dropout(encoded_feat)
        for i in range(self.n_layers):
            encoded_feat = getattr(self, 'fc%d' % i)(encoded_feat)
        logits = self.classifier(encoded_feat)
        return logits


class FullFCClassifierForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, feat_size, hidden_size, n_layers=6):
        super(FullFCClassifierForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, 'fc%d' % i, nn.Linear(feat_size, hidden_size))
            setattr(self, 'drop%d' % i, nn.Dropout(config.hidden_dropout_prob))
            setattr(self, 'cls%d' % i, nn.Linear(hidden_size, num_labels))
        self.apply(self.init_bert_weights)

    def forward(self, encoded_feat, weight, fc_index):
        logits = []
        for i, layer_idx in enumerate(fc_index):
            # print('feat.shape =', encoded_feat[layer_idx].shape)
            feat_drop = getattr(self, 'drop%d' % i)(encoded_feat[layer_idx])
            # print('feat_drop.shape =', feat_drop.shape)
            feat_fc = getattr(self, 'fc%d' % i)(feat_drop)
            # print('feat_fc.shape =', feat_fc.shape)
            logits.append(getattr(self, 'cls%d' % i)(feat_fc) * weight[i])
            # print('logits.shape =', logits[-1].shape)
            # print()

        logits = torch.stack(logits)
        #weight = weight.expand((logits.shape[2], logits.shape[1], logits.shape[0])).permute(2, 1, 0)
        #return (logits * weight).sum(0)
        return logits.sum(0)

