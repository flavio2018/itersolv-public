import logging
import torch


class Combiner(torch.nn.Module):

	def __init__(self):
		super(Combiner, self).__init__()

	def forward(self, expression, position, solution, sub_expressions_length):
		logging.info('Combiner forward.')
		new_expression = self._pool(expression, position, solution, sub_expressions_length)
		return new_expression

	def _is_final(self, solution):
		return solution == '$'

	def _pool(self, tokenized_expressions, positions, tokenized_solutions, sub_expressions_length):
		new_expressions = []
		for expression, position, solution, sub_exp_len in zip(tokenized_expressions, positions, tokenized_solutions, sub_expressions_length):
			solution_str = ''.join(solution)
			expression_str = ''.join(expression)
			if not self._is_final(solution_str):
				new_expressions.append(expression[:position] + solution + expression[position+sub_exp_len:])
			else:
				new_expressions.append(expression)
		new_expressions = [''.join(e) for e in new_expressions]
		return new_expressions