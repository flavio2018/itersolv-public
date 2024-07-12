import logging
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import torch


class VisualizeTaskMixin:

	def init_error_tables(self):
		self.errors_table = {}
		for dataset_name in self.datasets:
			if ('valid' in dataset_name) or ('test' in dataset_name):
				self.errors_table[dataset_name] = []

	@staticmethod
	def log_metrics_dict(metrics_dict):
		wandb.log(metrics_dict)

	@staticmethod
	def log_wandb_barplot(x, y, x_label, y_label):
		data = [[label, val] for (label, val) in zip(x, y)]
		table = wandb.Table(data=data, columns=[x_label, y_label])
		plot_id = '_'.join(y_label.split() + x_label.split())
		plot_title = f"{y_label} [{x_label}]"
		wandb.log({plot_id : wandb.plot.bar(table, x_label, y_label, title=plot_title)})

	def log_input_output(self, input, output, first_k=5):
		X_str = self.vocabulary.batch_to_str(input)
		out_str = self.vocabulary.batch_to_str(output, x=False)
		
		for x, y in zip(X_str[:first_k], out_str[:first_k]):
			logging.info(f"{x} → {y}")

	@staticmethod
	def log_wandb_table(rows, column_names, table_id):
		if rows:
			assert len(rows[0]) == len(column_names)
			
			table = wandb.Table(data=rows, columns=column_names)
			wandb.log({table_id: table})

	def store_errors_for_log(self, input, output, target, dataset_name, max_errors=2):
		assert len(input) == len(output) == len(target)

		if output.dim() == 3:
			output = output.argmax(-1)

		input_str = self.vocabulary.batch_to_str(input)
		output_str = self.vocabulary.batch_to_str(output, x=False)
		target_str = self.vocabulary.batch_to_str(target, x=False)

		count_added = 0
		for input, output, target in zip(input_str, output_str, target_str):
			output = self.vocabulary.cut_at_first_eos(output)
			if (output != target) and (count_added < max_errors):
				self.errors_table[dataset_name].append([input, output, target])
				count_added += 1
			elif count_added >= max_errors:
				return

	def add_weights_norm_to_metrics_dict(self):
		weights_norm = torch.norm(torch.concat([param.flatten() for param in self.model.parameters()]), 2)
		self.valid_step_metrics['model_stats/weights_norm'] = weights_norm
	
	def log_errors_table_end_run(self):
		for dataset_name, table in self.errors_table.items():
			self.log_wandb_table(table, column_names=['Input', 'Output', 'Target'], table_id=f"{dataset_name}/errors")

	def log_errors_table(self, dataset_name):
		self.log_wandb_table(self.errors_table[dataset_name], column_names=['Input', 'Output', 'Target'], table_id=f"{dataset_name}/errors")

	@staticmethod
	def log_heatmap(matrix, x_labels, y_labels, plot_id):
		fig, ax = plt.subplots()
		ax = sns.heatmap(
				data=matrix,
				xticklabels=x_labels,
				yticklabels=y_labels,
				square=True,
				ax=ax,
			)
		wandb.log({plot_id: wandb.Image(fig)})

	@staticmethod
	def log_lineplot(tensor, plot_id):
		x_len = len(tensor) if tensor.dim() == 1 else tensor.size(0)
		fig, ax = plt.subplots()
		ax = sns.lineplot(
				data=tensor,
			)
		ax.set_title(plot_id)
		ax.set_xticks(range(x_len), range(1, x_len+1))
		wandb.log({plot_id: wandb.Image(fig)})

	@staticmethod
	def log_scatterplot(dataframe, x, y, plot_id):
		fig, ax = plt.subplots()
		ax = sns.scatterplot(
				data=dataframe,
				x=x,
				y=y,
			)
		ax.set_title(plot_id)
		wandb.log({plot_id: wandb.Image(fig)})
	
	def log_n_model_params(self):
		n_model_params = sum(p.numel() for p in self.model.parameters())
		wandb.log({'model_stats/num_parameters': n_model_params})
		logging.info(f'Num. model parameters: {n_model_params}')
