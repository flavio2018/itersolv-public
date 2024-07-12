import os
import pickle


def get_run_filename(cfg):
	return f"{cfg.task_name}__nes{cfg.nesting}__nop{cfg.n_operands}__{cfg.prompt_type}__{cfg.batch_from}+{cfg.bs}"


def dump_run_tmp(cfg, zero_cot_first_outputs):
	print("Dumping zero_cot_first_outputs...", end=' ')
	with open(f"output/{cfg.model_name}/{get_run_filename(cfg)}.tmp", "wb") as tmp_f:
		pickle.dump(zero_cot_first_outputs, tmp_f)
	print("Done.")


def exists_run_tmp(cfg):
	return os.path.exists(f"output/{cfg.model_name}/{get_run_filename(cfg)}.tmp")


def load_run_tmp(cfg):
	with open(f"output/{cfg.model_name}/{get_run_filename(cfg)}.tmp", "rb") as tmp_f:
		zero_cot_first_outputs = pickle.load(tmp_f)
	return zero_cot_first_outputs