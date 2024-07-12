from collections import defaultdict
from dataclasses import dataclass
import torch
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import logging
from models.combiner import Combiner
from models.finder import Finder


class SelSolCom(torch.nn.Module):

	def __init__(self, selector, solver, vocabulary, n_multi, zoom_selector=False, length_threshold=None):
		super(SelSolCom, self).__init__()
		self.selector = selector
		self.solver = solver
		self.solver.vocabulary.sos = False
		self.finder = Finder(vocabulary)
		self.combiner = Combiner()
		self.n_multi = n_multi
		self.zoom_selector = zoom_selector
		self.length_threshold = length_threshold
		self.vocabulary = vocabulary
		if zoom_selector:
			assert self.n_multi % 10 == 0, "n_multi must be a multiple of 10 when zoom_selector=True."

	def forward(self, expression):
		batch_size = expression.size(0)
		is_valid = torch.ones(batch_size).bool().to(expression.device)
		is_final = torch.zeros(batch_size).bool().to(expression.device)
		self.run_batch_history = SelSolComBatchRunHistory(batch_size)
		self.track_stop_reason = {'sub_expression': torch.zeros(batch_size).to(expression.device),
								  'parentheses': torch.zeros(batch_size).to(expression.device)}
		self.count_solver_errors_per_seq = torch.zeros(batch_size).to(expression.device)
		self.hq_outputs_per_input = []  # (bs, solution steps)
		self.conf_score_by_input_len = defaultdict(list)

		while torch.bitwise_and(is_valid, ~is_final).any():
			expression_str = self.vocabulary.batch_to_str(expression[is_valid])
			self.run_batch_history.update_selector_inputs(expression_str, is_valid)
			logging.info('\n'.join(["Valid expressions:"] + expression_str))
			
			# get sub-expressions from expressions with a multi-run of the selector on the valid sequences
			sub_expression_obj = self._selector_multi_run_hq(expression[is_valid])
			
			count_hq = torch.full((batch_size,), np.nan).to(expression.device)
			count_hq[is_valid] = self.count_hq
			self.hq_outputs_per_input.append(count_hq)

			# make a batch of sub-expressions to be passed to the solver, keeping track of the order of valid sequences
			sub_expression_str = [o.text for o in sub_expression_obj]
			sub_expression_tmp = self.solver.vocabulary.str_to_batch(sub_expression_str)
			sub_expression = self.get_full_batch(sub_expression_tmp, is_valid, self.solver.vocabulary.get_special_idx('pad'))
			self.run_batch_history.update_selector_outputs(sub_expression_str, is_valid)

			# update the valid sequences based on the score given to the sub-expression by the finder
			is_valid = self._update_is_valid_using_finder_score(is_valid, sub_expression_obj)
			if ~is_valid.any():
				break

			# get the solutions of valid sub expressions only from the solver
			solution_tmp_obj = self._solver_multi_run_hq(sub_expression[is_valid])
			solution_tmp_str = [s.text for s in solution_tmp_obj]
			solution_tmp = self.solver.vocabulary.str_to_batch(solution_tmp_str, x=False)
			solution = self.get_full_batch(solution_tmp, is_valid, self.solver.vocabulary.get_special_idx('pad', x=False))
			self.run_batch_history.update_solver_outputs(solution_tmp_str, is_valid)

			# update the final sequences based on the output of the solver
			is_final = self._update_is_final(is_final, solution)

			# filter sub-expressions based on new valid sequences
			sub_expression_obj = [o for o in sub_expression_obj if o.normalized_finder_score >= 1]
			logging.info('\n'.join(["Valid sub-expressions:"] + [o.text for o in sub_expression_obj]))

			# prepare combiner inputs
			expression_str = self.vocabulary.batch_to_str(expression[is_valid])
			position = [o.position for o in sub_expression_obj]
			solution_str = self.solver.vocabulary.batch_to_str(solution[is_valid], x=False)
			solution_str = [s.replace(self.solver.vocabulary.specials['eos'], '') for s in solution_str]
			sub_expression_len = [len(o) for o in sub_expression_obj]
			logging.info('\n'.join(["Valid solutions:"] + solution_str))
			tokenized_expression = [self.vocabulary.tokenize_sample(e) for e in expression_str]
			tokenized_solution = [self.vocabulary.tokenize_sample(s) for s in solution_str]

			# run combiner to pool solutions into expressions
			new_expression_str = self.combiner(tokenized_expression, position, tokenized_solution, sub_expression_len)
			
			# update valid sequences based on the new expression
			is_valid = self._update_is_valid_using_expression(is_valid, new_expression_str)
			if ~is_valid.any():
				break
			
			# count errors due to solver mistakes
			self.update_solver_errors_counter(solution, sub_expression, is_valid)

			# filter new expression based on new valid sequences
			new_expression_str = [e for e in new_expression_str if self._has_well_formed_parentheses(e)]

			expression_tmp = self.selector.vocabulary.str_to_batch(new_expression_str)
			expression = self.get_full_batch(expression_tmp, is_valid, self.selector.vocabulary.get_special_idx('pad'))

		if ~is_valid.any():
			expression = self.get_full_batch(expression, is_valid, self.selector.vocabulary.get_special_idx('pad'))  # make a batch full of PAD
		
		expression = self.check_invalid_final_state(expression)
		expression_str = self.selector.vocabulary.batch_to_str(expression)
		expression_solve_vocab = self.vocabulary.str_to_batch(expression_str, x=False)
		return expression_solve_vocab
	
	def _zoom_in_selector_input(self, expression, length_threshold, zoom_perc):
		expression_lengths = (expression != self.selector.vocabulary.get_special_idx('pad')).sum(1)
		is_zoomed = expression_lengths > length_threshold

		if ~is_zoomed.any():
			return expression

		size_zoom_window = expression_lengths.detach().clone()
		size_zoom_window[is_zoomed] = (expression_lengths[is_zoomed] // (1 / zoom_perc)).to(torch.int64)

		list_zoomed_expressions = []

		for e, zoomed, win_size in zip(expression, is_zoomed, size_zoom_window):
			if zoomed:
				# expr_str = self.selector.vocabulary.batch_to_str(e.unsqueeze(0))[0]
				# zoomed_expr_str = expr_str[-win_size:]
				if (e == self.selector.vocabulary.get_special_idx('pad')).any():
					position_first_pad = (e == self.selector.vocabulary.get_special_idx('pad')).argwhere()[0].item()
					zoomed_expr_str = self.selector.vocabulary.batch_to_str(e[position_first_pad-win_size:position_first_pad].unsqueeze(0))[0]
				else:
					zoomed_expr_str = self.selector.vocabulary.batch_to_str(e[-win_size:].unsqueeze(0))[0]
				list_zoomed_expressions.append(zoomed_expr_str)
			else:
				expr_str = self.selector.vocabulary.batch_to_str(e.unsqueeze(0))[0]
				list_zoomed_expressions.append(expr_str)

		zoomed_expressions = self.selector.vocabulary.str_to_batch(list_zoomed_expressions)
		return zoomed_expressions

	def _selector_multi_run(self, expression):
		logging.info('Selector multi-run.')
		batch_size = expression.size(0)

		runs_outputs = { input_idx: [] for input_idx in range(batch_size) }
		
		for run_idx in range(self.n_multi):
			transformer_test_output = self.selector(expression)
			confidence_score = self._get_confidence_score(transformer_test_output)
			
			# convert sub-expression to expression vocabulary
			selector_output_str = self.selector.vocabulary.batch_to_str(transformer_test_output.tokens, x=False)
			selector_output_str = [o.replace(self.selector.vocabulary.specials['eos'], '') for o in selector_output_str]
			selector_output_str = [o.replace(self.selector.vocabulary.specials['hal'], '') for o in selector_output_str]
			selector_output_str = [o.replace(self.selector.vocabulary.specials['sos'], '') for o in selector_output_str]
			selector_output_solve_vocab = self.vocabulary.str_to_batch(selector_output_str)
			
			position, finder_score = self.finder(expression, selector_output_solve_vocab)
			
			for input_idx in range(batch_size):
				runs_outputs[input_idx].append(
					SelectorOutput(
						self.selector.vocabulary.tokenize_sample(selector_output_str[input_idx]),
						confidence_score[input_idx].item(),
						finder_score[input_idx].item(),
						position[input_idx].item())
					)

		for input_idx in runs_outputs.keys():
			runs_outputs[input_idx] = sorted(runs_outputs[input_idx], key=lambda o: (o.normalized_finder_score, o.confidence_score), reverse=True)
			logging.info('\n'.join(["5 best multi-run outputs:"] + [str(o) for o in runs_outputs[input_idx][:5]]))
			logging.info('\n'.join(["Unique multi-run outputs with perfect finder score:"] + list(set([o.text for o in runs_outputs[input_idx] if o.normalized_finder_score >= 1]))))

		best_outputs = [runs_outputs[input_idx][0] for input_idx in runs_outputs.keys()]
		return best_outputs
		
	def _selector_multi_run_hq(self, expression):
		logging.info('Selector multi-run.')
		batch_size = expression.size(0)
		expression_lengths = (expression != self.selector.vocabulary.get_special_idx('pad')).sum(1)

		runs_outputs = { input_idx: [] for input_idx in range(batch_size) }
		self.count_hq = torch.zeros(batch_size).to(expression.device)
		max_trials = self.n_multi

		if self.zoom_selector:
			repeats = self.n_multi // 20
			zoom_percentages = np.tile(np.linspace(0.1, 1, 10), repeats)
			zoom_percentages.sort()

		input_expression = expression.detach().clone()

		logging.info(f"Searching for HQ outputs...")
		for count_trials in range(max_trials):

			if self.zoom_selector:
				expression = self._zoom_in_selector_input(input_expression, length_threshold=self.length_threshold, zoom_perc=zoom_percentages[count_trials])

			transformer_test_output = self.selector(expression)
			confidence_score = self._get_confidence_score(transformer_test_output)
			proba_confidence_score = self._get_confidence_score(transformer_test_output, use_proba=True)
			
			# convert sub-expression to expression vocabulary
			selector_output_str = self.selector.vocabulary.batch_to_str(transformer_test_output.tokens, x=False)
			selector_output_str = [o.replace(self.selector.vocabulary.specials['eos'], '') for o in selector_output_str]
			selector_output_str = [o.replace(self.selector.vocabulary.specials['hal'], '') for o in selector_output_str]
			selector_output_str = [o.replace(self.selector.vocabulary.specials['sos'], '') for o in selector_output_str]
			selector_output_solve_vocab = self.vocabulary.str_to_batch(selector_output_str)
			
			position_finder, finder_score = self.finder(input_expression, selector_output_solve_vocab)
			
			for input_idx in range(batch_size):
				runs_outputs[input_idx].append(
					SelectorOutput(
						tokenized_text=self.selector.vocabulary.tokenize_sample(selector_output_str[input_idx]),
						confidence_score=confidence_score[input_idx].item(),
						finder_score=finder_score[input_idx].item(),
						position_finder=position_finder[input_idx].item())
					)

				if (runs_outputs[input_idx][-1].normalized_finder_score == 1):  # and (confidence_score[input_idx].item() >= -0.001):
					self.count_hq[input_idx] += 1

				self.conf_score_by_input_len[expression_lengths[input_idx].item()].append(proba_confidence_score[input_idx].item())

		logging.info(f"Run {count_trials+1} out of {max_trials} max trials. Found {self.count_hq.mean()} HQ outputs per input on average.")
		
		for input_idx in runs_outputs.keys():
			runs_outputs[input_idx] = sorted(runs_outputs[input_idx], key=lambda o: (o.normalized_finder_score, o.confidence_score), reverse=True)
			logging.info('\n'.join(["5 best multi-run outputs:"] + [str(o) for o in runs_outputs[input_idx][:5]]))
			logging.info('\n'.join(["Unique multi-run outputs with perfect finder score:"] + list(set([o.text for o in runs_outputs[input_idx] if o.normalized_finder_score >= 1]))))
		
		best_outputs = [runs_outputs[input_idx][0] for input_idx in runs_outputs.keys()]
		return best_outputs

	def _solver_multi_run_hq(self, sub_expression):
		logging.info("Solver multi-run.")
		batch_size = sub_expression.size(0)

		runs_outputs = { input_idx: [] for input_idx in range(batch_size) }
		is_hq = torch.zeros(batch_size).bool().to(sub_expression.device)
		count_trials, max_trials = 0, self.n_multi

		logging.info(f"Searching for HQ outputs...")
		while (not is_hq.all()) and (count_trials < max_trials):
			count_trials += 1
			transformer_test_output = self.solver(sub_expression)
			confidence_score = self._get_confidence_score(transformer_test_output)
			solver_output_str = self.solver.vocabulary.batch_to_str(transformer_test_output.tokens, x=False)		
			solver_output_str = [self.solver.vocabulary.cut_at_first_eos(o) for o in solver_output_str]
			
			for input_idx in range(batch_size):
				runs_outputs[input_idx].append(
					SolverOutput(
						solver_output_str[input_idx],
						confidence_score[input_idx].item(),
					)
				)

				if (confidence_score[input_idx].item() >= -0.001):
					is_hq[input_idx] = True

		logging.info(f"Run {count_trials} out of {max_trials} max trials. Found HQ outputs for {is_hq.sum()}/{batch_size} inputs.")
		
		for input_idx in runs_outputs.keys():
			runs_outputs[input_idx] = sorted(runs_outputs[input_idx], key=lambda o: o.confidence_score, reverse=True)
			logging.info('\n'.join(["5 best multi-run outputs:"] + [str(o) for o in runs_outputs[input_idx][:5]]))

		best_outputs = [runs_outputs[input_idx][0] for input_idx in runs_outputs.keys()]
		return best_outputs

	def get_full_batch(self, valid_batch, is_valid, filler):
		# make a batch keeping track of the order of valid sequences
		batch_size = len(is_valid)
		full_batch = torch.full((batch_size, valid_batch.size(1)), filler, device=valid_batch.device)
		if is_valid.any():
			full_batch[is_valid] = valid_batch
		return full_batch

	def _get_confidence_score(self, transformer_test_output, use_proba=False):
		# check exisis eos for all seq 
		exist_eos_in_seq = transformer_test_output.first_eos_mask.any(-1)
		
		# cover cases where no eos exists
		where_first_eos = torch.argwhere(transformer_test_output.first_eos_mask)[:, 1]
		where_first_eos_fixed = torch.zeros(transformer_test_output.tokens.size(0)).to(where_first_eos)
		where_first_eos_fixed[exist_eos_in_seq] = where_first_eos
		where_first_eos_fixed[~exist_eos_in_seq] = 2000  # some very big value
		
		# mask out positions after first eos
		positions_batch = torch.ones_like(transformer_test_output.tokens).cumsum(1) - 1
		positions_after_first_eos = positions_batch > where_first_eos_fixed.unsqueeze(1)
		transformer_test_output.proba[positions_after_first_eos] = 1
		
		# compute confidence score
		if use_proba:
			return torch.nan_to_num(transformer_test_output.proba.prod(-1), nan=0, posinf=0, neginf=0)
		else:
			output_log_proba = torch.log(transformer_test_output.proba + 1e-10)
			return torch.nan_to_num(output_log_proba.sum(-1), nan=0, posinf=0, neginf=0)

	def check_invalid_final_state(self, expression):
		open_parenthesis = '(' if self.selector.vocabulary.dataset_name != 'listops' else '['
		has_invalid_final_state = (expression[:, 0] == self.selector.vocabulary.x_vocab[open_parenthesis])
		if has_invalid_final_state.any():
			expression[has_invalid_final_state, :] = self.selector.vocabulary.get_special_idx('pad')
		for idx in has_invalid_final_state.argwhere():
			self.track_stop_reason['parentheses'][idx] += 1
		return expression

	def update_solver_errors_counter(self, solution, sub_expression, is_valid):
		valid_sub_expr = sub_expression[is_valid]
		valid_sub_expr_str = self.solver.vocabulary.batch_to_str(valid_sub_expr)
		valid_sub_expr_res = [self.eval_sub_expression(s_e) for s_e in valid_sub_expr_str]
		solution_str = self.solver.vocabulary.batch_to_str(solution[is_valid], x=False)
		solution_str = [s.replace(self.solver.vocabulary.specials['eos'], '') for s in solution_str]	
		solution_str = [s.replace('MIN', '').replace('MAX', '').replace('SM', '') for s in solution_str]
		if self.solver.vocabulary.dataset_name == 'algebra':
			solution_str = [parse_expr(s) % 100 if s != '$' else s for s in solution_str]
		pred_is_wrong = [true != pred for true, pred in zip(valid_sub_expr_res, solution_str)]
		self.count_solver_errors_per_seq[is_valid] += torch.tensor(pred_is_wrong).int().to(sub_expression.device)
		self.run_batch_history.update_solver_errors(valid_sub_expr_res, solution_str, is_valid)
	
	def _update_is_final(self, is_final, solution):
		is_final_now = solution[:, 0] == self.solver.vocabulary.get_special_idx('hal', x=False)
		is_final = torch.bitwise_or(is_final_now, is_final)
		return is_final

	def _update_is_valid_using_finder_score(self, is_valid, selector_multi_run_output):
		selector_output_it = iter(selector_multi_run_output)  # using iter since len(is_valid) != len(selector_multi_run_output) in general
		now_invalid_seq_idx = []
		now_invalid_seqs = []

		for sample_idx in range(len(is_valid)):
			if is_valid[sample_idx]:
				selector_output = next(selector_output_it)
				if selector_output.normalized_finder_score < 1:
					is_valid[sample_idx] = False
					now_invalid_seq_idx.append(sample_idx)
					now_invalid_seqs.append(selector_output.text)
		
		self.log_invalid_seq(now_invalid_seq_idx, now_invalid_seqs)
		
		return is_valid

	def _update_is_valid_using_expression(self, is_valid, expressions_str):
		expressions_it = iter(expressions_str)  # using iter since len(is_valid) != len(expressions_str) in general
		now_invalid_seq_idx = []
		now_invalid_seqs = []

		for sample_idx in range(len(is_valid)):
			if is_valid[sample_idx]:
				expression = next(expressions_it)
				if not self._has_well_formed_parentheses(expression):
					is_valid[sample_idx] = False
					now_invalid_seq_idx.append(sample_idx)
					now_invalid_seqs.append(expression)

		self.log_invalid_seq(now_invalid_seq_idx, now_invalid_seqs, sub_expression=False)
		
		return is_valid

	@staticmethod
	def _has_well_formed_parentheses(expression_str):
		if expression_str.count('[') != expression_str.count(']'):
			return False
		elif expression_str.count('(') != expression_str.count(')'):
			return False
		else:
			return True

	def log_invalid_seq(self, now_invalid_seq_idx, invalid_seqs, sub_expression=True):
		if sub_expression:
			reason = "the sub-expression was not found in the input"
			reason_short = 'sub_expression'
		else:
			reason = "the parentheses are not well-formed"
			reason_short = 'parentheses'

		num_invalid_seq = len(now_invalid_seq_idx)
		
		if num_invalid_seq > 0:
			logging.info(f"{num_invalid_seq} sequences are now invalid because {reason}:")
			logging.info('\n'.join(invalid_seqs))
			self.track_stop_reason[reason_short][now_invalid_seq_idx] += 1

	def eval_sub_expression(self, sub_expression):
		if self.solver.vocabulary.dataset_name == 'listops':
			tokenized_subexprssion = self.solver.vocabulary.tokenize_sample(sub_expression)
			args = [int(el) for el in tokenized_subexprssion if str.isdigit(el)]
			op = [el for el in tokenized_subexprssion if str.isalpha(el)]

			if len(op) > 0:
				op = op[0]
				if op == 'MIN':
					return str(min(args))
				elif op == 'MAX':
					return str(max(args))
				elif op == 'SM':
					return str(sum(args) % 10)
				else:
					assert False, 'Something went wrong somewhere.'
			else:
				return '$'

		elif self.solver.vocabulary.dataset_name == 'algebra':
			try:
				parsed_expr = parse_expr(sub_expression)
			except:
				print(f"Error while parsing {sub_expression}!")
				return -100
			if str(parsed_expr) == sub_expression:
				return '$'
			else:
				return parsed_expr % 100

		else:
			if not '(' in sub_expression:
				return '$'
			else:
				try:
					value = eval(sub_expression)
					if value < 0:
						value_mod = value % -100
					else:
						value_mod = value % 100
					return str(value_mod)
				except SyntaxError:
					return str(-1000)


@dataclass(frozen=True)
class SelectorOutput:
	tokenized_text: str
	confidence_score: float
	finder_score: float
	position_finder: int

	def __len__(self):
		return len(self.tokenized_text)

	@property
	def text(self):
		return ''.join(self.tokenized_text)

	@property
	def normalized_finder_score(self):
		if len(self.tokenized_text) == 0:
			return 0
		return self.finder_score / len(self.tokenized_text)

	@property
	def position(self):
		return self.position_finder
	

@dataclass(frozen=True)
class SolverOutput:
	text: str
	confidence_score: float


class SelSolComBatchRunHistory:

	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.selector_inputs = {k: [] for k in range(batch_size)}
		self.selector_outputs = {k: [] for k in range(batch_size)}
		self.solver_outputs = {k: [] for k in range(batch_size)}
		self.solver_errors = {k: [] for k in range(batch_size)}

	def update_selector_inputs(self, selector_inputs_curr_iter, is_valid):
		valid_idx_iterator = iter(is_valid.argwhere().flatten().tolist())
		for selector_input in selector_inputs_curr_iter:
			valid_idx = next(valid_idx_iterator)
			self.selector_inputs[valid_idx].append(selector_input)

	def update_selector_outputs(self, selector_outputs_curr_iter, is_valid):
		valid_idx_iterator = iter(is_valid.argwhere().flatten().tolist())
		for selector_output in selector_outputs_curr_iter:
			valid_idx = next(valid_idx_iterator)
			self.selector_outputs[valid_idx].append(selector_output)

	def update_solver_outputs(self, solver_outputs_curr_iter, is_valid):
		valid_idx_iterator = iter(is_valid.argwhere().flatten().tolist())
		for solver_output in solver_outputs_curr_iter:
			valid_idx = next(valid_idx_iterator)
			self.solver_outputs[valid_idx].append(solver_output)

	def update_solver_errors(self, true, pred, is_valid):
		valid_idx_iterator = iter(is_valid.argwhere().flatten().tolist())
		for t, p in zip(true, pred):
			valid_idx = next(valid_idx_iterator)
			if t != p:
				self.solver_errors[valid_idx].append((t, p))
