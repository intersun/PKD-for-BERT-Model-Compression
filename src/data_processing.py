from src.nli_data_processing import init_glue_model, get_glue_task_dataloader
from src.race_data_processing import init_race_model, get_race_task_dataloader


def init_model(task_name, output_all_layers, num_hidden_layers, config):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model(task_name, output_all_layers, num_hidden_layers, config)


def get_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    if 'race' in task_name.lower():
        return get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)
    else:
        return get_glue_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)

