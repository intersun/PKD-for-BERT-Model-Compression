import logging
import os
import random
import pickle

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from BERT.pytorch_pretrained_bert.modeling import BertConfig
from BERT.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer

from src.argument_parser import default_parser, get_predefine_argv, complete_argument
from src.nli_data_processing import processors, output_modes
from src.data_processing import init_model, get_task_dataloader
from src.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from src.utils import load_model, count_parameters, eval_model_dataloader_nli, eval_model_dataloader
from src.KD_loss import distillation_loss, patience_loss
from envs import HOME_DATA_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#########################################################################
# Prepare Parser
##########################################################################
parser = default_parser()
DEBUG = True
if DEBUG:
    logger.info("IN DEBUG MODE")
    # run simple fune-tuning *teacher* by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'finetune_teacher')

    # run simple fune-tuning *student* by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'finetune_student')

    # run vanilla KD by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'kd')

    # run Patient Teacher by uncommenting below cmd
    # argv = get_predefine_argv('glue', 'RTE', 'kd.cls')
    try:
        args = parser.parse_args(argv)
    except NameError:
        raise ValueError('please uncomment one of option above to start training')
else:
    logger.info("IN CMD MODE")
    args = parser.parse_args()
args = complete_argument(args)

args.raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', args.task_name)
args.feat_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat', args.task_name)

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
logger.info('actual batch size on all GPU = %d' % args.train_batch_size)
device, n_gpu = args.device, args.n_gpu

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Prepare  Data
##########################################################################
task_name = args.task_name.lower()

if task_name not in processors and 'race' not in task_name:
    raise ValueError("Task not found: %s" % (task_name))

if 'race' in task_name:
    pass
else:
    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

if args.do_train:
    train_sampler = SequentialSampler if DEBUG else RandomSampler
    read_set = 'train'
    if args.teacher_prediction is not None and args.alpha > 0:
        logger.info('loading teacher\'s predictoin')
        teacher_predictions = pickle.load(open(args.teacher_prediction, 'rb'))['train'] if args.teacher_prediction is not None else None
        logger.info('teacher acc = %.2f, teacher loss = %.5f' % (teacher_predictions['acc']*100, teacher_predictions['loss']))
        if args.kd_model == 'kd':
            train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                                      batch_size=args.train_batch_size,
                                                                      knowledge=teacher_predictions['pred_logit'])
        else:
            train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                                      batch_size=args.train_batch_size,
                                                                      knowledge=teacher_predictions['pred_logit'],
                                                                      extra_knowledge=teacher_predictions['feature_maps'])

    else:
        if args.alpha > 0:
            raise ValueError('please specify teacher\'s prediction file for KD training')
        logger.info('runing simple fine-tuning because teacher\'s prediction is not provided')
        train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                                  batch_size=args.train_batch_size)
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Run prediction for full data
    eval_examples, eval_dataloader, eval_label_ids = get_task_dataloader(task_name, 'dev', tokenizer, args, SequentialSampler, batch_size=args.eval_batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

if args.do_eval:
    test_examples, test_dataloader, test_label_ids = get_task_dataloader(task_name, 'test', tokenizer, args, SequentialSampler, batch_size=args.eval_batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)


#########################################################################
# Prepare model
#########################################################################
student_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
if args.kd_model.lower() in ['kd', 'kd.cls']:
    logger.info('using normal Knowledge Distillation')
    output_all_layers = args.kd_model.lower() == 'kd.cls'
    student_encoder, student_classifier = init_model(task_name, output_all_layers, args.student_hidden_layers, student_config)

    n_student_layer = len(student_encoder.bert.encoder.layer)
    student_encoder = load_model(student_encoder, args.encoder_checkpoint, args, 'student', verbose=True)
    logger.info('*' * 77)
    student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'classifier', verbose=True)
elif args.kd_model.lower() == 'kd.full':
    logger.info('using FULL Knowledge Distillation')
    layer_idx = [int(i) for i in args.fc_layer_idx.split(',')]
    num_fc_layer = len(layer_idx)
    if args.weights is None or args.weights.lower() in ['none']:
        weights = np.array([1] * (num_fc_layer-1) + [num_fc_layer-1]) / 2 / (num_fc_layer-1)
    else:
        weights = [float(w) for w in args.weights.split(',')]
        weights = np.array(weights) / sum(weights)

    assert len(weights) == num_fc_layer, 'number of weights and number of FC layer must be equal to each other'

    # weights = torch.tensor(np.array([1, 1, 1, 1, 2, 6])/12, dtype=torch.float, device=device, requires_grad=False)
    # if args.fp16:
    #    weights = weights.half()
    student_encoder = BertForSequenceClassificationEncoder(student_config, output_all_encoded_layers=True,
                                                           num_hidden_layers=args.student_hidden_layers,
                                                           fix_pooler=True)
    n_student_layer = len(student_encoder.bert.encoder.layer)
    student_encoder = load_model(student_encoder, args.encoder_checkpoint, args, 'student', verbose=True)
    logger.info('*' * 77)

    student_classifier = FullFCClassifierForSequenceClassification(student_config, num_labels, student_config.hidden_size,
                                                                   student_config.hidden_size, 6)
    student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'exact', verbose=True)
    assert max(layer_idx) <= n_student_layer - 1, 'selected FC layer idx cannot exceed the number of transformers'
else:
    raise ValueError('%s KD not found, please use kd or kd.full' % args.kd)

n_param_student = count_parameters(student_encoder) + count_parameters(student_classifier)
logger.info('number of layers in student model = %d' % n_student_layer)
logger.info('num parameters in student model are %d and %d' % (count_parameters(student_encoder), count_parameters(student_classifier)))


#########################################################################
# Prepare optimizer
#########################################################################
if args.do_train:
    param_optimizer = list(student_encoder.named_parameters()) + list(student_classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        logger.info('FP16 activate, use apex FusedAdam')
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        logger.info('FP16 is not activated, use BertAdam')
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)


#########################################################################
# Model Training
#########################################################################
output_model_file = '{}_nlayer.{}_lr.{}_T.{}.alpha.{}_beta.{}_bs.{}'.format(args.task_name, args.student_hidden_layers,
                                                                            args.learning_rate,
                                                                            args.T, args.alpha, args.beta,
                                                                            args.train_batch_size * args.gradient_accumulation_steps)
if args.do_train:
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    student_encoder.train()
    student_classifier.train()

    log_train = open(os.path.join(args.output_dir, 'train_log.txt'), 'w', buffering=1)
    log_eval = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w', buffering=1)
    print('epoch,global_steps,step,acc,loss,kd_loss,ce_loss,AT_loss', file=log_train)
    print('epoch,acc,loss', file=log_eval)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_ce_loss, tr_kd_loss, tr_acc = 0, 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            if args.alpha == 0:
                input_ids, input_mask, segment_ids, label_ids = batch
                teacher_pred, teacher_patience = None, None
            else:
                if args.kd_model == 'kd':
                    input_ids, input_mask, segment_ids, label_ids, teacher_pred = batch
                    teacher_patience = None
                else:
                    input_ids, input_mask, segment_ids, label_ids, teacher_pred, teacher_patience = batch
                    if args.fp16:
                        teacher_patience = teacher_patience.half()
                if args.fp16:
                    teacher_pred = teacher_pred.half()

            # define a new function to compute loss values for both output_modes
            full_output, pooled_output = student_encoder(input_ids, segment_ids, input_mask)
            if args.kd_model.lower() in['kd', 'kd.cls']:
                logits_pred_student = student_classifier(pooled_output)
                if args.kd_model.lower() == 'kd.cls':
                    student_patience = torch.stack(full_output[:-1]).transpose(0, 1)
                else:
                    student_patience = None
            elif args.kd_model.lower() == 'kd.full':
                logits_pred_student = student_classifier(full_output, weights, layer_idx)
            else:
                raise ValueError(f'{args.kd_model} not implemented yet')

            loss_dl, kd_loss, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha=args.alpha)
            if args.beta > 0:
                if student_patience.shape[0] != input_ids.shape[0]:
                    # For RACE
                    n_layer = student_patience.shape[1]
                    student_patience = student_patience.transpose(0, 1).contiguous().view(n_layer, input_ids.shape[0], -1).transpose(0,1)
                pt_loss = args.beta * patience_loss(teacher_patience, student_patience, args.normalize_patience)
                loss = loss_dl + pt_loss
            else:
                pt_loss = torch.tensor(0.0)
                loss = loss_dl

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            n_sample = input_ids.shape[0]
            tr_loss += loss.item() * n_sample
            if isinstance(kd_loss, float):
                tr_kd_loss += kd_loss * n_sample
            else:
                tr_kd_loss += kd_loss.item() * n_sample
            tr_ce_loss += ce_loss.item() * n_sample
            tr_loss_pt = pt_loss.item() * n_sample

            pred_cls = logits_pred_student.data.max(1)[1]
            tr_acc += pred_cls.eq(label_ids).sum().cpu().item()
            nb_tr_examples += n_sample
            nb_tr_steps += 1

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.log_every_step == 0:
                print('{},{},{},{},{},{},{},{}'.format(epoch+1, global_step, step, tr_acc / nb_tr_examples,
                                                       tr_loss / nb_tr_examples, tr_kd_loss / nb_tr_examples,
                                                       tr_ce_loss / nb_tr_examples, tr_loss_pt / nb_tr_examples),
                      file=log_train)

        # Save a trained model and the associated configuration
        if 'race' in task_name:
            result = eval_model_dataloader(student_encoder, student_classifier, eval_dataloader, device, False)
        else:
            result = eval_model_dataloader_nli(args.task_name.lower(), eval_label_ids, student_encoder, student_classifier, eval_dataloader,
                                               args.kd_model, num_labels, device, args.weights, args.fc_layer_idx, output_mode)
        if args.task_name in ['CoLA']:
            print('{},{},{}'.format(epoch+1, result['mcc'], result['eval_loss']), file=log_eval)
        else:
            if 'race' in args.task_name:
                print('{},{},{}'.format(epoch+1, result['acc'], result['loss']), file=log_eval)
            else:
                print('{},{},{}'.format(epoch+1, result['acc'], result['eval_loss']), file=log_eval)

        if args.n_gpu > 1:
            torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.encoder.pkl'))
            torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.cls.pkl'))
        else:
            torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.encoder.pkl'))
            torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.cls.pkl'))

if args.do_eval:
    if 'race' not in args.task_name:
        result = eval_model_dataloader_nli(args.task_name.lower(), test_label_ids, student_encoder, student_classifier, test_dataloader,
                                           args.kd_model, num_labels, device, args.weights, args.fc_layer_idx, output_mode)
    else:
        result = eval_model_dataloader(student_encoder, student_classifier, test_dataloader, device, False)

    output_test_file = os.path.join(args.output_dir, "test_results_" + output_model_file + '.txt')
    with open(output_test_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
# Mismatch are deleted for now
