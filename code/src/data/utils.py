from data.dataset import ItersolvDataset


def build_dataset(data_cfg, split, device, difficulty_split=None):
	return ItersolvDataset(
			dataset_name=data_cfg.name,
			split=split,
			train_batch_size=data_cfg.train_batch_size,
			eval_batch_size=data_cfg.eval_batch_size,
			device=device,
			sos=data_cfg.sos,
			eos=data_cfg.eos,
			difficulty_split=difficulty_split,
			specials_in_x=data_cfg.specials_in_x)