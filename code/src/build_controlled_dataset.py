#!/usr/bin/env python


import hydra
import pandas as pd
import utils
import pickle


@hydra.main(config_path="../conf/", version_base='1.2')
def main(cfg):
	gen = utils.get_generator(cfg)

	final_df, samples_to_serialize = build_set(gen, cfg, cfg[cfg.dataset_name].difficulty_splits)
	
	splits_counts = final_df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
	print(splits_counts)

	final_df.to_csv(f'../datasets/{cfg.dataset_name}_controlled.csv', index=False)
	with open(f'../datasets/{cfg.dataset_name}_controlled_ood_samples.pickle', 'wb') as pickle_f:
		pickle.dump(samples_to_serialize, pickle_f)
	utils.dump_config(cfg, f"../datasets/")


def build_set(generator, cfg, splits_parameters):
	samples = set()
	samples_to_serialize = []

	for split_parameters in splits_parameters:
		print(f"Difficulty split: {split_parameters}.")
		split_samples = set()
		
		if len(split_parameters) == 2:	# standard param spec
			nesting, num_operands = split_parameters
			extra = 'no'
			tasks = ['select', 'solve']
			generator.easy = cfg.easy
		
		elif len(split_parameters) == 3:  # listops train param spec
			nesting, num_operands, extra = split_parameters
			
			generator.easy = True if extra == 'easy' else cfg.easy
			if extra == 'step':
				tasks = ['select_step', 'solve']
			elif extra in ['s1', 's2', 's3', 's4']:
				tasks = [f'select_{extra}', 'solve']
			else:
				tasks = ['select', 'solve']

		counter = 0
		count_samples = 0

		if split_parameters in cfg[cfg.dataset_name].dev_splits.train:
			max_samples = cfg.max_samples_for_training_splits
		else:
			max_samples = cfg.max_samples_for_eval_splits

		if split_parameters in cfg[cfg.dataset_name].dev_splits.valid_ood:
			store_samples_for_serialization = True
		else:
			store_samples_for_serialization = False

		while (counter < cfg.max_wait) and (len(split_samples) < max_samples):
			Xs, Ys = generator.generate_samples(10, nesting=nesting, num_operands=num_operands, split=None, exact=True, task=tasks)

			for x_select, x_solve, y_select, y_solve in zip(*Xs, *Ys):
				if (len(split_samples) < max_samples):
					split_samples.add((x_select, x_solve, y_select, y_solve, nesting, num_operands, extra))

			if len(split_samples) > count_samples:
				count_samples = len(split_samples)
				counter = 0
			else:
				counter += 1
			print(f"{len(split_samples):6d} samples | counter: {counter:4d}.", end='\r')

			if store_samples_for_serialization:
				samples_to_serialize += generator.samples

		samples |= split_samples
		print()
	
	df_dict = {
		'X_select': [sample[0] for sample in samples],
		'X_solve': [sample[1] for sample in samples],
		'Y_select': [sample[2] for sample in samples],
		'Y_solve': [sample[3] for sample in samples],
		'nesting': [sample[4] for sample in samples],
		'num_operands': [sample[5] for sample in samples],
		'extra': [sample[6] for sample in samples],
	}

	df = pd.DataFrame(df_dict)
	print(f"Num unique samples: {len(df)}.")
	print(f"{len(samples_to_serialize)} samples to serialize.")
	return df, samples_to_serialize


if __name__ == '__main__':
	main()
