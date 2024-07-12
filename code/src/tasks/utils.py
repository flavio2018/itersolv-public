from models.utils import build_transformer, build_selsolcom
from data.utils import build_dataset
from tasks.train import TrainTask
from tasks.test import TestSSCTask


def build_task(cfg):
    datasets = dict()
    
    for split in cfg.data.splits:
        datasets[split] = build_dataset(cfg.data, split, cfg.device)
    
    if 'difficulty_splits' in cfg.data:
        for nesting, num_operands in cfg.data.difficulty_splits:
            datasets[f"test_{nesting}_{num_operands}"] = build_dataset(cfg.data, 'test', cfg.device, (nesting, num_operands))

    if 'test_selsolcom' in cfg.name:
        vocabularies = dict()
        vocabularies['selector'] = build_dataset(cfg.selector_data, 'train', cfg.device).vocabulary
        vocabularies['solver'] = build_dataset(cfg.solver_data, 'train', cfg.device).vocabulary
        vocabularies['selsolcom'] = datasets['test'].vocabulary
    else:
        vocab_split = 'train' if 'train' in cfg.name else 'test'
        print(f"Building vocab with {vocab_split} dataset split.")
        vocabulary = datasets[vocab_split].vocabulary

    if cfg.model.name == 'transformer':
        model = build_transformer(cfg.model, cfg.device, vocabulary)
    elif cfg.model.name == 'selsolcom':
        model = build_selsolcom(cfg.model, cfg.device, vocabularies)

    if 'train' in cfg.name:
        task = TrainTask(model, datasets, cfg)

    elif 'test' in cfg.name:
        task = TestSSCTask(model, datasets, cfg)

    return task