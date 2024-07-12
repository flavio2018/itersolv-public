import logging
import omegaconf
import os
from data.generators import AlgebraicExpressionGenerator, ArithmeticExpressionGenerator, ListOpsExpressionGenerator


def get_generator(cfg):
	if cfg.dataset_name == 'algebra':
		return AlgebraicExpressionGenerator(
			mini_steps=cfg.algebra.mini_steps,
			modulo=cfg.algebra.modulo,
			variables=cfg.algebra.variables,
			coeff_variables=cfg.algebra.coeff_variables,
			easy=cfg.easy)
	elif cfg.dataset_name == 'arithmetic':
		return ArithmeticExpressionGenerator(
			mini_steps=cfg.arithmetic.mini_steps,
			modulo=cfg.arithmetic.modulo,
			min_operand_value=cfg.arithmetic.min_operand_value,
			max_operand_value=cfg.arithmetic.max_operand_value,
			operators=cfg.arithmetic.operators,
			easy=cfg.easy)
	elif cfg.dataset_name == 'listops':
		return ListOpsExpressionGenerator(
			mini_steps=cfg.listops.mini_steps,
			simplify_last=cfg.listops.simplify_last,
			ops=cfg.listops.ops,
			easy=cfg.easy)


def dump_config(cfg, dir):
	with open(f'{dir}config.txt', 'w') as f:
		f.write(omegaconf.OmegaConf.to_yaml(cfg))


def make_dir_if_not_exists(cfg):
	easy = '_easy' if cfg.easy else ''
	if not os.path.exists(f'../datasets/{cfg.dataset_name}{cfg.variant_name}_{cfg.task}{easy}/'):
		os.mkdir(f'../datasets/{cfg.dataset_name}{cfg.variant_name}_{cfg.task}{easy}/')


def mirror_logging_to_console():
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(message)s')
	console.setFormatter(formatter)
	logging.getLogger().addHandler(console)
