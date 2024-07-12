#!/usr/bin/env python3


import glob
import functools
import pandas as pd
from collections import Counter
from parse import build_parser
import utils
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application


def main():
	model_name = 'gpt4'
	task_names = ['listops', 'arithmetic', 'algebra']

	nesting_ranges = {
		'arithmetic': range(1, 7),
		'algebra': range(1, 7),
		'listops': range(1, 5),
	}

	num_operands_ranges = {
		'arithmetic': range(2, 3),
		'algebra': range(2, 3),
		'listops': range(2, 5)
	}

	for task_name in task_names:
		if task_name == 'listops':
			accuracy_table = pd.DataFrame(index=[1, 2, 3, 4], columns=[2, 3, 4])
		else:
			accuracy_table = pd.DataFrame(index=[1, 2, 3, 4], columns=[2])

		for nesting in nesting_ranges[task_name]:
			for num_operands in num_operands_ranges[task_name]:
				output_files = glob.glob(f'output/{model_name}/{task_name}__nes{nesting}__nop{num_operands}__self_consistency*.csv')
				output_files = [o for o in output_files if 'merged' not in o]

				if len(output_files) >= 1:
					print(f"Evaluating self_consistency for {task_name}, ({nesting}, {num_operands}).")
					
					most_common_outputs = get_most_common_outputs(output_files)
					df = pd.read_csv(output_files[0], index_col=0)
					df['parsed_output'] = most_common_outputs
					# df.to_csv(f'output/{model_name}/{task_name}/self_consistency/{task_name}__nes{nesting}__nop{num_operands}__self_consistency__merged.csv')
					run_acc = eval_df(df)
					accuracy_table.loc[nesting, num_operands] = run_acc

				else:
					accuracy_table.loc[nesting, num_operands] = -1

		accuracy_table.to_csv(f'output/accuracy_tables/{model_name}_{task_name}_self_consistency.csv')


	for task_name in task_names:
		if task_name == 'listops':
			accuracy_table = pd.DataFrame(index=[1, 2, 3, 4], columns=[2, 3, 4])
		else:
			accuracy_table = pd.DataFrame(index=[1, 2, 3, 4], columns=[2])

		for nesting in nesting_ranges[task_name]:
			for num_operands in num_operands_ranges[task_name]:
				output_files = glob.glob(f'output/{model_name}/{task_name}__nes{nesting}__nop{num_operands}__self_consistency*.csv')
				output_files = [o for o in output_files if 'merged' not in o]

				print(f"Evaluating zero-shot CoT (1 output) for {task_name}, ({nesting}, {num_operands}).")
				print(f"Reading file {output_files[0]}")
				df = pd.read_csv(output_files[0], index_col=0)
				parser = build_parser(df['task_name'][0])
				add_parsed_output_to_df(df, parser)
				run_acc = eval_df(df)
				accuracy_table.loc[nesting, num_operands] = run_acc

		accuracy_table.to_csv(f'output/accuracy_tables/{model_name}_{task_name}_zero_shot_cot.csv')


def get_most_common_outputs(output_files):
	parsed_outputs = []

	for file in output_files:
		print(f"Reading file {file}")
		df = pd.read_csv(file, index_col=0)
		parser = build_parser(df['task_name'][0])
		add_parsed_output_to_df(df, parser)		
		parsed_outputs.append(df['parsed_output'].tolist())
		print(f"{parser.error_counter} errors while parsing file.")

	parsed_outputs_T = [o for o in zip(*parsed_outputs)]
	most_common_outputs = [Counter(outputs).most_common()[0][0] for outputs in parsed_outputs_T]
	return most_common_outputs


def add_parsed_output_to_df(df, parser):
	if (df['task_name'][0] == 'arithmetic') or (df['task_name'][0] == 'listops'):
		try:
			df['gpt_output'] = df['gpt_output'].apply(str).apply(lambda s: s.replace('.0', '').replace('.', '')).apply(lambda x: int(x))
		except ValueError as err:
			print(err)
		df['gpt_output'] = df['gpt_output'].astype(str)
	else:
		df['gpt_output'] = df['gpt_output'].astype(str)

	df['parsed_output'] = df['gpt_output'].apply(parser.parse_outputs)
	
	if df['task_name'][0] == 'algebra':
		df['parsed_output'] = df['parsed_output'].apply(expr_to_sympy_w_except)


def eval_df(df):
	if df['task_name'][0] == 'algebra':
		return eval_sym_df(df)
	else:
		return eval_str_df(df)


def eval_str_df(run_df):
	assert 'parsed_output' in run_df
	return (run_df['original_target'] == run_df['parsed_output']).mean()


def eval_sym_df(run_df):
	assert 'parsed_output' in run_df
	return (run_df['original_target'].apply(expr_to_sympy_w_except) == run_df['parsed_output']).mean()


def expr_to_sympy_w_except(expr):
	transformations = (standard_transformations + (implicit_multiplication_application,))
	parse_expr_part = functools.partial(parse_expr, transformations=transformations)
	
	try:
		parsed_expr = parse_expr_part(expr)
	except SyntaxError as err:
		print(f"Could not parse symbolic expression: {expr}")
		parsed_expr = parse_expr_part('-100')
	return parsed_expr


if __name__ == '__main__':
	main()
