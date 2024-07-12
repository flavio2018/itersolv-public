#!/usr/bin/env python


import hydra
import pandas as pd
import utils
import logging


@hydra.main(config_path="../conf", version_base='1.2')
def main(cfg):
	utils.make_dir_if_not_exists(cfg)
	gen = utils.get_generator(cfg)
	
	logging.basicConfig(level=logging.INFO,
	                    format='%(asctime)s\n%(message)s',
	                    datefmt='%y-%m-%d %H:%M',
	                    filename=f'../datasets/{cfg.dataset_name}_{cfg.task}/run.log',
	                    filemode='w')

	utils.mirror_logging_to_console()
	
	df = build_samples_set(gen, cfg, cfg[cfg.dataset_name].difficulty_splits)
	num_train_samples = round(cfg.num_train_samples[cfg.dataset_name] * len(df))
	logging.info(f"{num_train_samples} train samples.")

	df = df.sample(frac=1)  # shuffle df

	train_df = df.iloc[:num_train_samples, :]

	splits_counts = train_df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
	logging.info(splits_counts)

	logging.info(f"Upsampling training set.")
	train_df = upsample_training_set(train_df)
	logging.info(f"{len(train_df)} train samples.")

	splits_counts = train_df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
	logging.info(splits_counts)
	
	train_df.to_csv(f'../datasets/{cfg.dataset_name}_{cfg.task}/train.csv', index=False)
	df.iloc[num_train_samples:, :].to_csv(f'../datasets/{cfg.dataset_name}_{cfg.task}/valid_iid.csv', index=False)

	utils.dump_config(cfg, f"../datasets/{cfg.dataset_name}_{cfg.task}/")


def build_samples_set(generator, cfg, splits_parameters):
	SAMPLES_PER_BATCH = 10
	samples = set()
	max_range = 5000

	for split_parameters in splits_parameters:
		logging.info(f"Difficulty split {split_parameters}")
		split_samples = set()

		if len(split_parameters) == 2:	# standard param spec
			nesting, num_operands = split_parameters
			task = cfg.task
			extra = 'no'
		elif len(split_parameters) == 3:  # listops train param spec
			nesting, num_operands, extra = split_parameters
			if extra == 'no_par':
				task = cfg.task + '_no_par'
		
		if num_operands < 2:
			generator.easy = True
		else:
			generator.easy = cfg.easy

		for it in range(max_range):
			X, Y = generator.generate_samples(SAMPLES_PER_BATCH, nesting=nesting, num_operands=num_operands, split=None, exact=True, task=[task])
			for x, y in zip(X, Y):
				split_samples.add((x, y, nesting, num_operands, extra))

			if (cfg.max_samples_per_split[cfg.dataset_name] is not None) and (len(split_samples) > cfg.max_samples_per_split[cfg.dataset_name]):
				break

			print(f"{len(split_samples):6d} samples | it: {it+1:4d}.", end='\r')
		
		samples |= split_samples
		print()

	df_dict = {
		'X': [sample[0] for sample in samples],
		'Y': [sample[1] for sample in samples],
		'nesting': [sample[2] for sample in samples],
		'num_operands': [sample[3] for sample in samples],
		'extra': [sample[4] for sample in samples],
	}
	
	total = len(df_dict['X'])
	logging.info(f"Num unique samples: {total}.")
	
	df = pd.DataFrame(df_dict)
	return df


def upsample_training_set(df):
	splits_counts = df.groupby(['nesting', 'num_operands', 'extra'], dropna=False).count()
	max_count = splits_counts['X'].max()

	for split in splits_counts.index:
		split_count = splits_counts.loc[split, 'X']
		if split_count != max_count:
			nesting, num_operands, extra = split
			upsample_factor = max_count//split_count - 1
			split_samples = df[(df['nesting']==nesting) & (df['num_operands']==num_operands) & (df['extra']==extra)]
			df = pd.concat([df] + [split_samples]*upsample_factor)

	return df


if __name__ == '__main__':
	main()