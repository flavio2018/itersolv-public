#!/usr/bin/env python


import hydra
import logging
import os
import pickle
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import utils


@hydra.main(config_path="../conf/", version_base='1.2')
def main(cfg):
	setup_folders(cfg)
	setup_logging(cfg)
	logging.info(cfg)
	
	logging.info("Loading dataframe...")
	df = pd.read_csv(f"../datasets/{cfg.dataset_name}_controlled.csv")
	logging.info(f"Dataframe loaded. {len(df)} samples.")

	logging.info("Loading pickled OOD samples...")
	with open(f'../datasets/{cfg.dataset_name}_controlled_ood_samples.pickle', 'rb') as pickle_f:
		serialized_ood_samples = pickle.load(pickle_f)
	logging.info(f"Loaded {len(serialized_ood_samples)} samples.")
	
	splits_counts = df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
	logging.info(splits_counts)

	dev_splits = make_dev_splits(df, cfg)

	dev_splits_dfs = {dev_split_name: pd.concat(dfs) for dev_split_name, dfs in dev_splits.items()}
	
	dump_ndr_datasets(dev_splits_dfs, cfg)

	dev_splits_dfs['valid_ood'] = expand_valid_ood_set(cfg, dev_splits_dfs['valid_ood'], serialized_ood_samples)

	log_dev_splits_stats(dev_splits_dfs)
	
	dev_splits_dfs['train'] = upsample_training_set(dev_splits_dfs['train'])
	
	dump_task_datasets(dev_splits_dfs, cfg)


def setup_folders(cfg):
	if not os.path.exists(f'../datasets/{cfg.dataset_name}_controlled_select/'):
		os.mkdir(f'../datasets/{cfg.dataset_name}_controlled_select/')
	
	if not os.path.exists(f'../datasets/{cfg.dataset_name}_controlled_solve/'):
		os.mkdir(f'../datasets/{cfg.dataset_name}_controlled_solve/')

	if not os.path.exists(f'../datasets/{cfg.dataset_name}_controlled_ndr/'):
		os.mkdir(f'../datasets/{cfg.dataset_name}_controlled_ndr/')


def setup_logging(cfg):
	logging.basicConfig(level=logging.DEBUG,
	                    format='%(asctime)s\n%(message)s',
	                    datefmt='%y-%m-%d %H:%M',
	                    filename=f'../datasets/{cfg.dataset_name}_controlled_select/run.log',
	                    filemode='w')
	utils.mirror_logging_to_console()


def make_dev_splits(df, cfg):
	samples = {
		"train": [],
		"valid_iid": [],
		"valid_ood": [],
		"test": [],
	}

	for difficulty_split, dev_splits_qty in cfg.splits.items():
		nes, op, ex = eval(difficulty_split)
		df_difficulty_split = df[(df['nesting'] == nes) & (df['num_operands'] == op) & (df['extra'] == ex)]
		indices = get_np_split_indices(dev_splits_qty, df_difficulty_split)
		df_dev_splits = np.split(df_difficulty_split, indices)
		df_dev_splits = df_dev_splits[:-1]  # due to how np.split works
		logging.info(f"Difficulty split {difficulty_split} was divided in {len(df_dev_splits)} dev splits.")

		non_empty_splits = [k for k, v in dev_splits_qty.items() if v != 0]

		for dev_split_name, dev_split_samples in zip(non_empty_splits, df_dev_splits):
			samples[dev_split_name].append(dev_split_samples)

	return samples


def get_np_split_indices(dev_splits_qty, data):
	qty = dev_splits_qty.values()
	qty = filter_nonzero(qty)
	qty = make_cumulative(qty)
	
	if isinstance(dev_splits_qty['train'], float):
		indices = [int(perc*len(data)) for perc in qty]
	else:
		indices = qty
	
	return indices


def filter_nonzero(qty):
	return [q for q in qty if q != 0]


def make_cumulative(qty):
	return np.cumsum(qty)


def expand_valid_ood_set(cfg, ood_samples_df, serialized_ood_samples):
	logging.info(f"{len(ood_samples_df)} basic OOD samples.")

	sample_str2obj = {s.to_string(): s for s in serialized_ood_samples}
	max_samples_per_statistic = cfg.max_ood_samples_per_split
	samples_by_statistic = defaultdict(list)

	for sample_str in ood_samples_df['X_select']:
		sample_obj = sample_str2obj[sample_str]
		
		for depth, subexpr_ops, x, y in sample_obj.get_solution_chain_stats():
			samples_by_statistic[(depth, subexpr_ops)].append((x, y))
	
	for statistic, samples in samples_by_statistic.items():
		samples_by_statistic[statistic] = random.sample(samples, max_samples_per_statistic)

	logging.info(f"{len(samples_by_statistic)} statistics in expanded OOD set.")

	expanded_ood_samples = {
		'X': [],
		'Y': [],
		'depth': [],
		'subexpr_ops': [],
	}

	for (depth, subexpr_ops), samples in samples_by_statistic.items():
		for x, y in samples:
			expanded_ood_samples['X'].append(x)
			expanded_ood_samples['Y'].append(y)
			expanded_ood_samples['depth'].append(depth)
			expanded_ood_samples['subexpr_ops'].append(subexpr_ops)

	expanded_ood_samples_df = pd.DataFrame(expanded_ood_samples)
	expanded_ood_samples_df = expanded_ood_samples_df.drop(np.where(expanded_ood_samples_df['subexpr_ops'] == 0)[0])
	expanded_ood_samples_df = expanded_ood_samples_df.drop(np.where(expanded_ood_samples_df['depth'] == 0)[0])

	logging.info(f"{len(expanded_ood_samples_df)} samples in expanded OOD set.")
	
	logging.info(expanded_ood_samples_df.groupby(['depth', 'subexpr_ops']).count())

	return expanded_ood_samples_df


def log_dev_splits_stats(samples_dfs):
	for dev_split_name, samples_df in samples_dfs.items():
		logging.info('')
		logging.info(f"{dev_split_name}, {len(samples_df)} samples.")
		if dev_split_name != 'valid_ood':
			splits_counts = samples_df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
			logging.info(splits_counts)



def upsample_training_set(train_df):
	logging.info("Upsampling training set")
	
	splits_counts = train_df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
	max_count = splits_counts['X_select'].max()

	for split in splits_counts.index:
		split_count = splits_counts.loc[split, 'X_select']
		if split_count != max_count:
			nesting, num_operands, extra = split
			upsample_factor = max_count//split_count - 1
			split_samples = train_df[(train_df['nesting']==nesting) & (train_df['num_operands']==num_operands) & (train_df['extra']==extra)]
			train_df = pd.concat([train_df] + [split_samples]*upsample_factor)

	logging.info(f"{len(train_df)} samples.")
	logging.info(train_df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count())

	return train_df


def dump_task_datasets(samples_dfs, cfg):
	for dev_split_name, samples_df in samples_dfs.items():
		(samples_df.loc[:, samples_df.columns.difference(['Y_solve', 'X_solve'])]
				   .rename(columns={'Y_select': 'Y', 'X_select': 'X'})
				   .to_csv(f"../datasets/{cfg.dataset_name}_controlled_select/{dev_split_name}.csv", index=False))

		if dev_split_name == 'test':	# also dump solve dataset
			(samples_df.loc[:, samples_df.columns.difference(['Y_select', 'X_select'])]
					   .rename(columns={'Y_solve': 'Y', 'X_solve': 'X'})
					   .to_csv(f"../datasets/{cfg.dataset_name}_controlled_solve/{dev_split_name}.csv", index=False))


def dump_ndr_datasets(samples_dfs, cfg):
	for dev_split_name, samples_df in samples_dfs.items():
		samples_df['Y_ndr'] = samples_df['Y_solve']
		samples_df.loc[(samples_df['nesting']==1) & (samples_df['num_operands']==1), 'Y_ndr'] = samples_df[(samples_df['nesting']==1) & (samples_df['num_operands']==1)]['X_solve']
		
		if dev_split_name in ['train', 'valid_iid', 'valid_ood']:
			samples_df = samples_df[samples_df['extra'] == 'no']
			(samples_df.loc[:, samples_df.columns.difference(['X_select', 'Y_select', 'Y_solve'])]
			  		   .rename(columns={'Y_ndr': 'Y', 'X_solve': 'X'})
			  		   .to_csv(f"../datasets/{cfg.dataset_name}_controlled_ndr/{dev_split_name}.csv", index=False))
	
	valid_df = pd.concat([samples_dfs['valid_iid'], samples_dfs['valid_ood']])
	valid_df = valid_df[valid_df['extra'] == 'no']
	(valid_df.loc[:, valid_df.columns.difference(['X_select', 'Y_select', 'Y_solve'])]
			  		   .rename(columns={'Y_ndr': 'Y', 'X_solve': 'X'})
			  		   .to_csv(f"../datasets/{cfg.dataset_name}_controlled_ndr/valid.csv", index=False))

	for dev_split_name, samples_df in samples_dfs.items():
		samples_df.drop('Y_ndr', axis=1, inplace=True)
	

if __name__ == '__main__':
	main()
