import pandas as pd
import re


def build_input_target(task_name, batch_from, bs, nesting, n_ops):
	df = pd.read_csv(f'../datasets/{task_name}_controlled_solve/test.csv')
	df = df.loc[(df['nesting'] == nesting) & (df['num_operands'] == n_ops)]
	assert len(df) > 0, "No samples found!"

	if task_name == 'listops':
		df['X'] = df['X'].apply(reformat_listops_expression)
	
	batch_to = batch_from + bs
	batch = df['X'].tolist()[batch_from:batch_to]
	target = df['Y'].tolist()[batch_from:batch_to]
	return batch, target


def reformat_listops_expression(expr):
	listops_re = re.compile(r'(\d)|(SM|MIN|MAX)|([\[\]])|([?.#$])')
	matches = listops_re.findall(expr)
	expr_w_spaces = " ".join([[submatch for submatch in match if submatch][0] for match in matches])
	return expr_w_spaces.replace('[ ', '[').replace(' ]', ']')
