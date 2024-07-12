from collections import defaultdict
import logging
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from tasks.base import BaseTask
from tasks.mixins import EvalTaskMixin, VisualizeTaskMixin
from utils import mirror_logging_to_console


class TestSSCTask(BaseTask, VisualizeTaskMixin, EvalTaskMixin):

	def __init__(self, model, dataset, cfg):
		super(TestSSCTask, self).__init__(model, dataset, cfg)
		mirror_logging_to_console()


	def _load_ckpt(self, opt=True):
		assert self.model is not None

		if self.cfg.model.solver.ckpt:
			self.model.solver.load_model_weights(self.cfg.model.solver.ckpt)
	
	def run(self):
		super().run()
		self.test()

	def test(self):
		self.init_error_tables()
		self.test_fine_grained()
		# self.test_aggregate()

	def test_aggregate(self):
		logging.info("Running aggregate test.")
		char_acc, seq_acc = self.test_model_on_split('test')
		self.log_metrics_dict({
				'char_acc/total': char_acc,
				'seq_acc/total': seq_acc,
				'stop_reason/sub_expression/total': self.perc_stopped_due_to_sub_expression,
				'stop_reason/parentheses/total': self.perc_stopped_due_to_parentheses,
			})

	def test_fine_grained(self):
		logging.info("Running fine-grained test.")
		char_acc_matrix = self.get_matrix_for_heatmap(self.cfg.data.difficulty_splits)
		seq_acc_matrix = self.get_matrix_for_heatmap(self.cfg.data.difficulty_splits)
		self.reset_count_hq_outputs_per_input(self.cfg.data.difficulty_splits)
		self.reset_conf_score_by_input_len()
		self.reset_error_analysis_df()

		for nesting, num_operands in self.cfg.data.difficulty_splits:
			split_name = f"test_{nesting}_{num_operands}"
			char_acc, seq_acc = self.test_model_on_split(split_name)
			char_acc_matrix[nesting-1, num_operands-1] = char_acc
			seq_acc_matrix[nesting-1, num_operands-1] = seq_acc
			self.log_metrics_dict({
					f'char_acc/N{nesting}_O{num_operands}': char_acc,
					f'seq_acc/N{nesting}_O{num_operands}': seq_acc,
					f'stop_reason/sub_expression/N{nesting}_O{num_operands}': self.perc_stopped_due_to_sub_expression,
					f'stop_reason/parentheses/N{nesting}_O{num_operands}': self.perc_stopped_due_to_parentheses,
					f'stop_reason/solver/N{nesting}_O{num_operands}': self.perc_mistakes_only_due_to_solver,
					f'avg_solver_err_seq/N{nesting}_O{num_operands}': self.avg_solver_errors_per_seq,
				})
			self.log_errors_table(f"test_{nesting}_{num_operands}")
			self.update_error_analysis_df(f'N{nesting}_O{num_operands}', [self.perc_stopped_due_to_sub_expression.cpu().item(),
																		  self.perc_stopped_due_to_parentheses.cpu().item(),
																		  self.perc_mistakes_only_due_to_solver.cpu().item()])

		self.log_lineplot(self.avg_hq_outputs_per_input, 'hq_outputs/all')
		self.log_scatterplot(self.conf_score_by_input_len_df, 'input_len', 'avg_conf_score', 'input_len_Vs_conf_scores')
		self.log_heatmap(char_acc_matrix, range(1, num_operands+1), range(1, nesting+1), 'char_acc/heatmap')
		self.log_heatmap(seq_acc_matrix, range(1, num_operands+1), range(1, nesting+1), 'seq_acc/heatmap')
		self.dump_confidence_scores_df()
		self.dump_error_analysis_df()
	
	@torch.no_grad()
	def test_model_on_split(self, split_name):
		logging.info(f"Testing model on split {split_name}")
		self.model.eval()
		char_acc_values = []
		seq_acc_values = []
		total_it = len(self.dataloaders[split_name]) // self.cfg.data.eval_batch_size
		self.reset_track_stop_reason()
		self.reset_split_run_history()
		self.reset_track_true_pred()

		for X, Y in tqdm(iter(self.dataloaders[split_name]), total=total_it):
			output = self.model(X)
			output = self._fix_output_shape(output, Y)
			self.update_track_stop_reason(self.model)
			self.update_split_run_history(self.model.run_batch_history)
			self.update_track_true_pred(Y, output)
			self.update_count_hq_outputs_per_inputs(self.model, split_name)
			self.update_conf_score_by_input_len(self.model)
			char_acc_values += [self.batch_acc_no_reduce(output, Y)]
			seq_acc_values += [self.batch_seq_acc_no_reduce(output, Y)]
			self.store_errors_for_log(X, output, Y, split_name, max_errors=100)

		self.pred_is_exact = torch.concat(seq_acc_values).flatten()
		char_acc = torch.concat(char_acc_values).flatten().mean()
		seq_acc = self.pred_is_exact.mean()
		return char_acc, seq_acc

	def reset_track_stop_reason(self):
		self.track_stop_reason = {'sub_expression': [], 'parentheses': []}
		self.count_solver_errors = []

	def reset_split_run_history(self):
		self.split_run_history = {"selector_inputs": [],
								  "selector_outputs": [],
								  "solver_outputs": [],
								  "solver_errors": [],}

	def reset_track_true_pred(self):
		self.track_true_pred = {'true': [], 'pred': []}

	def reset_count_hq_outputs_per_input(self, difficulty_splits):
		self.count_hq_outputs_per_input = {f"N{n}_O{o}": [] for n, o in difficulty_splits}

	def reset_conf_score_by_input_len(self):
		self.conf_score_by_input_len = defaultdict(list)

	def reset_error_analysis_df(self):
		self.error_analysis_df = pd.DataFrame(columns=['sub_expression', 'parentheses', 'solver'])

	def update_track_stop_reason(self, model):
		for reason, seq_has_stopped in model.track_stop_reason.items():
			self.track_stop_reason[reason].append(seq_has_stopped)
		self.count_solver_errors.append(model.count_solver_errors_per_seq)

	def update_split_run_history(self, ssc_batch_run_history):
		self.split_run_history['selector_inputs'] += ssc_batch_run_history.selector_inputs.values()
		self.split_run_history['selector_outputs'] += ssc_batch_run_history.selector_outputs.values()
		self.split_run_history['solver_outputs'] += ssc_batch_run_history.solver_outputs.values()
		self.split_run_history['solver_errors'] += ssc_batch_run_history.solver_errors.values()

	def update_track_true_pred(self, true, pred):
		self.track_true_pred['true'].append(self.vocabulary.batch_to_str(true, x=False))
		self.track_true_pred['pred'].append(self.vocabulary.batch_to_str(pred, x=False))

	def update_count_hq_outputs_per_inputs(self, model, split_name):
		if split_name == 'test':  # skip on whole test set
			return
		nesting, num_operands = split_name[5:].split('_')
		batch_count = torch.concat([hq.unsqueeze(0) for hq in model.hq_outputs_per_input]).T
		batch_count = batch_count.tolist()
		self.count_hq_outputs_per_input[f"N{nesting}_O{num_operands}"] += batch_count

	def update_conf_score_by_input_len(self, model):
		for input_len, scores in model.conf_score_by_input_len.items():
			self.conf_score_by_input_len[input_len] += scores

	def update_error_analysis_df(self, split_idx, row):
		self.error_analysis_df.loc[split_idx] = row

	def dump_confidence_scores_df(self):
		self.conf_score_by_input_len_df.to_csv(f'../out/{self.cfg.data.name}_{self.cfg.model.n_multi}_input_len_Vs_conf_scores.csv', index=False)

	def dump_error_analysis_df(self):
		self.error_analysis_df.to_csv(f'../out/{self.cfg.data.name}_{self.cfg.model.n_multi}_error_analysis.csv')

	@property
	def avg_hq_outputs_per_input(self):
		formatted_matrix = []
		for count in self.count_hq_outputs_per_input.values():
			max_num_outputs = max(len(c) for c in count)
			formatted = []
			for count_hq_outputs in count:
				base = torch.full((max_num_outputs,), np.nan).to(self.model.solver.device)
				base[:len(count_hq_outputs)] = 0
				base[:len(count_hq_outputs)] += torch.tensor(count_hq_outputs, device=self.model.solver.device)
				formatted.append(base)
			formatted_matrix.append(torch.concat([f.unsqueeze(0) for f in formatted]).nanmean(0).cpu())
		
		max_num_iters = max(len(f) for f in formatted_matrix)
		base_formatted = torch.full((len(self.count_hq_outputs_per_input), max_num_iters), np.nan).to(self.model.solver.device)
		for vec_idx, formatted_vector in enumerate(formatted_matrix):
			base_formatted[vec_idx, :len(formatted_vector)] = 0
			base_formatted[vec_idx, :len(formatted_vector)] = formatted_vector
		
		return base_formatted.T.cpu()

	def print_run_history_seq(self, seq_idx):
		for sel_in, sel_out, sol_out in zip(self.split_run_history['selector_inputs'][seq_idx], self.split_run_history['selector_outputs'][seq_idx], self.split_run_history['solver_outputs'][seq_idx]):
			print(sel_in, sel_out, sol_out)

	@property
	def stopped_due_to_sub_expression(self):
		return torch.concat(self.track_stop_reason['sub_expression']).flatten()

	@property
	def perc_stopped_due_to_sub_expression(self):
		return self.stopped_due_to_sub_expression.mean()

	@property
	def stopped_due_to_parentheses(self):
		return torch.concat(self.track_stop_reason['parentheses']).flatten()

	@property
	def perc_stopped_due_to_parentheses(self):
		return self.stopped_due_to_parentheses.mean()

	@property
	def mistakes_only_due_to_solver(self):
		count_solver_errors = torch.concat(self.count_solver_errors).flatten()
		seq_has_solver_error = count_solver_errors > 0
		seq_wrong_out_solver_err = torch.bitwise_and(seq_has_solver_error, ~self.pred_is_exact.bool().to(seq_has_solver_error.device))
		return torch.bitwise_and(torch.bitwise_and(seq_wrong_out_solver_err, ~self.stopped_due_to_parentheses.bool()), ~self.stopped_due_to_sub_expression.bool())
		
	@property
	def perc_mistakes_only_due_to_solver(self):
		return self.mistakes_only_due_to_solver.float().mean()

	@property
	def avg_solver_errors_per_seq(self):
		return torch.concat(self.count_solver_errors).flatten().mean()
	
	@property
	def conf_score_by_input_len_df(self):
		df_dict = {
			'input_len': [],
			'avg_conf_score': [],
		}
		for input_len, scores in self.conf_score_by_input_len.items():
			# if input_len > 5
			df_dict['input_len'].append(input_len)
			df_dict['avg_conf_score'].append(torch.tensor(scores).mean().item())
		df = pd.DataFrame(df_dict)
		df = df.sort_values(by='input_len')
		return df

	@staticmethod
	def get_matrix_for_heatmap(difficulty_splits):
		max_nesting, max_num_operands = 0, 0
		for nesting, num_operands in difficulty_splits:
			if nesting > max_nesting:
				max_nesting = nesting
			if num_operands > max_num_operands:
				max_num_operands = num_operands
		return np.zeros((max_nesting, max_num_operands))
