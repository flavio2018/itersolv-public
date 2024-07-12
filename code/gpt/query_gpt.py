#!/usr/bin/env python

from data import build_input_target
from prompts import get_prompt_builder
from openai_interface import OpenAIInterface
import pandas as pd
import hydra
import datetime
import utils


@hydra.main(config_path="../conf/", config_name="query_gpt", version_base='1.2')
def main(cfg):
	print(f"##### {cfg.task_name} nesting {cfg.nesting}, {cfg.n_operands} operands, {cfg.prompt_type} prompt. #####")

	run_timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
	batch, target = build_input_target(
		**{
			"task_name": cfg.task_name,
			"batch_from": cfg.batch_from,
			"bs": cfg.bs,
			"nesting": cfg.nesting,
			"n_ops": cfg.n_operands
		}
	)
	
	prompt_builder = get_prompt_builder(cfg.task_name)
	prompts = prompt_builder.build_prompt(batch, cfg.prompt_type)
	openai_interface = OpenAIInterface(cfg, run_timestamp)

	outputs = openai_interface.query_model_zero_shot_cot(prompts)
	zero_cot_first_outputs = openai_interface.zero_cot_first_outputs

	print("Dumping run DataFrame...")
	df = pd.DataFrame(
	    columns=["task_name",
	             "prompt_type",
	             "original_input",
	             "prompt",
	             "0_shot_cot_first_out",
	             "gpt_output",
	             "original_target"]
	)

	df['prompt'] = prompts 
	df['original_input'] = batch
	df['original_target'] = target
	df['task_name'] = cfg.task_name
	df['prompt_type'] = cfg.prompt_type
	df['gpt_output'] = outputs
	df['0_shot_cot_first_out'] = zero_cot_first_outputs

	df.to_csv(f'output/{cfg.model_name}/{utils.get_run_filename(cfg)}__{run_timestamp}.csv')
	print("Done.")


if __name__ == '__main__':
	main()
