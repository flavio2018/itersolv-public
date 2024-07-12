import torch
import warnings
import datetime as dt


class EvalTaskMixin:

	def reset_metrics_dict(self):
		self.valid_step_metrics = dict()

	def valid_step(self, iteration):
		self.valid_step_metrics['iteration'] = self.restored_run_final_it + iteration
		self.add_weights_norm_to_metrics_dict()
		start_valid_timestamp = dt.datetime.now()

		for dataset_name in self.datasets:
			if 'valid' in dataset_name:
				self.valid_model_on_set(dataset_name)

		end_valid_timestamp = dt.datetime.now()
		self.valid_step_metrics['train_stats/valid_duration'] = (end_valid_timestamp - start_valid_timestamp).total_seconds()
		self.valid_step_metrics['train_stats/eta'] = self.eta(iteration)
		self.valid_step_metrics['train_stats/training_throughput'] = self.training_throughput
		self.valid_step_metrics['lr'] = self.scheduler.get_last_lr()[-1] if self.cfg.task.lr_scheduler is not None else self.cfg.task.lr
		self.log_metrics_dict(self.valid_step_metrics)

	@torch.no_grad()
	def valid_model_on_set(self, set_name):
		self.model.eval()
		char_acc_values = []
		seq_acc_values = []
		cumulative_loss, sum_valid = 0, 0

		for X, Y in iter(self.dataloaders[set_name]):
			output = self.model(X, Y[:, :-1], tf=self.tf)
			char_acc_values += [self.batch_acc_no_reduce(output, Y[:, 1:])]
			seq_acc_values += [self.batch_seq_acc_no_reduce(output, Y[:, 1:])]
			cum_loss_it, sum_valid_it = self.compute_loss_no_reduce(output, Y[:, 1:])
			cumulative_loss += cum_loss_it
			sum_valid += sum_valid_it
		
		loss = self.reduce_loss(cumulative_loss, sum_valid)
		char_acc = torch.concat(char_acc_values).flatten().mean()
		seq_acc = torch.concat(seq_acc_values).flatten().mean()

		self.valid_step_metrics[f'{set_name}/loss'] = loss.item()
		self.valid_step_metrics[f'{set_name}/char_acc'] = char_acc.item()
		self.valid_step_metrics[f'{set_name}/seq_acc'] = seq_acc.item()

		self.store_errors_for_log(X, output, Y[:, 1:], set_name)

	def batch_acc_no_reduce(self, outputs, targets):
		idx_pad = self.vocabulary.get_special_idx('pad', x=False)
		mask = (targets != idx_pad).to(outputs.device)
		if outputs.dim() == 2:
			idx_outs = outputs
		else:	# assuming outputs.dim() == 3 -- i.e. (bs, seq_len, vocab_dim)
			idx_outs = outputs.argmax(dim=-1)
		out_equal_target = (idx_outs == targets).type(torch.FloatTensor).to(outputs.device)
		valid_out_equal_target = torch.masked_select(out_equal_target, mask)
		return valid_out_equal_target

	def batch_acc(self, outputs, targets):
		valid_out_equal_target = self.batch_acc_no_reduce(outputs, targets)
		return valid_out_equal_target.mean(), valid_out_equal_target.std()
	
	def batch_seq_acc_no_reduce(self, outputs, targets):
		idx_pad = self.vocabulary.get_special_idx('pad', x=False)
		mask = (targets != idx_pad).type(torch.int32)
		len_Y = mask.sum(dim=-1)
		if outputs.dim() == 2:
			idx_outs = outputs
		else:	# assuming outputs.dim() == 3 -- i.e. (bs, seq_len, vocab_dim)
			idx_outs = outputs.argmax(dim=-1)
		out_equal_target = (idx_outs == targets).type(torch.int32)
		masked_out_equal_target = out_equal_target * mask
		num_equal_chars_per_seq = masked_out_equal_target.sum(dim=-1)
		pred_is_exact = (num_equal_chars_per_seq == len_Y).type(torch.FloatTensor)
		return pred_is_exact

	def batch_seq_acc(self, outputs, targets):
		pred_is_exact = self.batch_seq_acc_no_reduce(outputs, targets)
		return pred_is_exact.mean(), pred_is_exact.std()

	def batch_output_in_input_acc(self, outputs, inputs):
		seq_acc_multi_xattn = torch.zeros(outputs.size(0), device=outputs.device)
		outputs_str = self.vocabulary.batch_to_str(outputs, x=False)
		inputs_str = self.vocabulary.batch_to_str(inputs, x=False)

		for seq_idx, (o, i) in enumerate(zip(outputs_str, inputs_str)):
			o = o.split(self.vocabulary.specials['sep'])[0]
			if o in i:
				seq_acc_multi_xattn[seq_idx] = 1
		return seq_acc_multi_xattn.mean(), seq_acc_multi_xattn.std()

	def compute_loss(self, outputs, targets):
		cumulative_loss, sum_valid = self.compute_loss_no_reduce(outputs, targets)
		return self.reduce_loss(cumulative_loss, sum_valid)

	def compute_loss_no_reduce(self, outputs, targets):
		cumulative_loss = 0
		idx_pad = self.vocabulary.get_special_idx('pad', x=False)
		mask = (targets != idx_pad).type(torch.int32)
		for char_pos in range(outputs.size(1)):
			char_loss = self.criterion(outputs[:, char_pos, :], targets[:, char_pos])
			masked_char_loss = char_loss * mask[:, char_pos]
			cumulative_loss += masked_char_loss.sum()
		return cumulative_loss, mask.sum()

	def reduce_loss(self, cumulative_loss, sum_valid):
		return cumulative_loss / sum_valid

	def _fix_output_shape(self, output, Y):
		# fix pred/target shape mismatch
		if output.size(1) < Y.size(1):
			warnings.warn(f"Outputs batch shape {output.size()} is different from targets batch shape {Y.size()}. Fixing.")
			missing_timesteps = Y.size(1) - output.size(1)
			pad_tokens = torch.tensor(self.vocabulary.get_special_idx('pad', x=False), device=Y.device).tile(output.size(0), missing_timesteps)
			output = torch.concat([output, pad_tokens], dim=1)
		elif output.size(1) > Y.size(1):
			warnings.warn(f"Outputs batch shape {output.size()} is different from targets batch shape {Y.size()}. Fixing.")
			output = output[:, :Y.size(1)]
		return output
	