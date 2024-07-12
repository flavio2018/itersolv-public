import numpy as np
import hydra
import logging
import os
import datetime as dt
import torch
from collections import deque
from tasks.base import BaseTask
from tasks.mixins import EvalTaskMixin, VisualizeTaskMixin


class TrainTask(BaseTask, VisualizeTaskMixin, EvalTaskMixin):

	def __init__(self, model, dataset, cfg):
		super(TrainTask, self).__init__(model, dataset, cfg)
		self.restored_run_final_it = 0
		self.start_timestamp = dt.datetime.now()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.task.lr)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
		self.use_early_stopping = self.cfg.task.use_early_stop
		self.best_valid_score = None
		if self.use_early_stopping:
			self._init_early_stop()
		else:
			self.early_stopping = None
		if self.cfg.model.ckpt:
			self._load_state_from_ckpt()
		if self.cfg.task.lr_scheduler is not None:
			schedulers = {
				'step': torch.optim.lr_scheduler.StepLR(self.opt, step_size=cfg.task.lr_scheduler_step),
				'cyclic': torch.optim.lr_scheduler.CyclicLR(self.opt,
					base_lr=cfg.task.cyclic_scheduler_base_lr,
					max_lr=cfg.task.cyclic_scheduler_max_lr,
					step_size_up=cfg.task.cyclic_scheduler_step_up,
					step_size_down=cfg.task.cyclic_scheduler_step_down,
					cycle_momentum=False,
					last_epoch=self.get_cyclic_scheduler_last_epoch()),
				'cosann': torch.optim.lr_scheduler.SequentialLR(self.opt,
					schedulers=[torch.optim.lr_scheduler.LinearLR(self.opt,
									start_factor=cfg.task.linear_scheduler_start_factor,
									total_iters=cfg.task.linear_scheduler_iters,
									last_epoch=self.get_linear_scheduler_last_epoch()),
								torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,
									cfg.task.cosann_max_iters,
									eta_min=cfg.task.linear_scheduler_start_factor*cfg.task.lr,
									last_epoch=self.get_cosann_scheduler_last_epoch())],
					milestones=self.get_sequential_scheduler_milestones(),
					last_epoch=self.get_sequential_scheduler_last_epoch()),
			}
			self.scheduler = schedulers[cfg.task.lr_scheduler]
			if cfg.task.lr_scheduler == 'cosann':
				self.scheduler._schedulers[1]._step_count = 0
		self.reset_metrics_dict()

	def eta(self, it):
		if it < 500:
			return None
		elapsed_time = (dt.datetime.now() - self.start_timestamp).total_seconds()
		estimated_time_per_iter = elapsed_time / (it + 1)
		remaining_iters = self.cfg.task.max_iter - it + 1
		return (remaining_iters * estimated_time_per_iter) / 60

	@property
	def training_throughput(self):
		last_train_step_duration = (self.end_train_step_timestamp - self.start_train_step_timestamp).total_seconds()
		bs = self.dataloaders['train'].dataset.train_batch_size
		return bs / last_train_step_duration

	def run(self):
		super().run()
		self.init_error_tables()
		self._set_tf()
		self.train()
		self.log_errors_table_end_run()

	def train(self):
		for it in range(self.cfg.task.max_iter):
			self.train_step()

			if it % self.FREQ_WANDB_LOG == 0:
				self.valid_step(it)
				self.serialize_state(self.valid_step_metrics[self.cfg.task.early_stop_metric], it)
				if self.use_early_stopping and self.early_stopping.early_stop:
					return
				self.reset_metrics_dict()

	def train_step(self):
		self.start_train_step_timestamp = dt.datetime.now()
		self.model.train()
		self.opt.zero_grad()
		X, Y = next(iter(self.dataloaders['train']))
		outputs = self.model(X, Y[:, :-1], tf=self.tf)
		loss = self.compute_loss(outputs, Y[:, 1:])
		acc, std = self.batch_acc(outputs, Y[:, 1:])
		seq_acc, std = self.batch_seq_acc(outputs, Y[:, 1:])
		loss.backward()
		self.opt.step()
		if self.cfg.task.lr_scheduler is not None:
			self.scheduler.step()
		self.valid_step_metrics['train/loss'] = loss.item()
		self.valid_step_metrics['train/char_acc'] = acc.item()
		self.valid_step_metrics['train/seq_acc'] = seq_acc.item()
		self.end_train_step_timestamp = dt.datetime.now()

	def _load_state_from_ckpt(self):
		ckpt = torch.load(
				os.path.join(hydra.utils.get_original_cwd(),
					f'../checkpoints/{self.cfg.model.ckpt}'), map_location=self.cfg.device)
		self._load_opt_from_ckpt(ckpt)
		self._set_restored_run_final_it(ckpt)
		
	def _load_opt_from_ckpt(self, ckpt):
		logging.info('Loading optimizer from checkpoint...')
		self.opt.load_state_dict(ckpt['opt'])
		logging.info('Done.')

	def _set_restored_run_final_it(self, ckpt):
		self.restored_run_final_it = ckpt['update']

	def get_cyclic_scheduler_last_epoch(self):
		# assuming cyclic lr scheduler parameters have not changed wrt ckpt
		if self.restored_run_final_it == 0:
			return -1
		else:
			return self.restored_run_final_it % (self.cfg.task.cyclic_scheduler_step_up + self.cfg.task.cyclic_scheduler_step_up)

	def get_linear_scheduler_last_epoch(self):
		if self.restored_run_final_it == 0:
			return -1
		else:
			return self.restored_run_final_it

	def get_cosann_scheduler_last_epoch(self):
		if self.restored_run_final_it == 0:
			return -1
		else:
			return self.restored_run_final_it

	def get_sequential_scheduler_last_epoch(self):
		if self.restored_run_final_it == 0:
			return -1
		else:
			return self.restored_run_final_it

	def get_sequential_scheduler_milestones(self):
		if self.restored_run_final_it == 0:
			return [self.cfg.task.linear_scheduler_iters]
		else:
			return [0]

	def _init_early_stop(self):
		goal_maximize = True if 'acc' in self.cfg.task.early_stop_metric else False
		self.early_stopping = EarlyStopping(
			patience=self.cfg.task.early_stop_patience,
			verbose=True,
			path=os.path.join(hydra.utils.get_original_cwd(),
			 f"../checkpoints/{self.cfg.start_timestamp}_{self.cfg.name}.pth"),
			trace_func=logging.info,
			goal_maximize=goal_maximize,
			model_cfg=self.cfg.model,
			restored_run_final_it=self.restored_run_final_it,
		)

	def _set_tf(self):
		if isinstance(self.cfg.task.tf, float):
			self.tf = True if (torch.rand(1) > self.cfg.task.tf).item() else False
		else:
			self.tf = self.cfg.task.tf

	def serialize_state(self, val_loss_step, it):
		if self.use_early_stopping:
			self.early_stopping(val_loss_step, self.model, self.opt, it)
		else:
			self._simple_serialize_state(val_loss_step, it)

	def _simple_serialize_state(self, loss_step, it):
		goal_maximize = True if 'acc' in self.cfg.task.early_stop_metric else False
		if goal_maximize:
			has_improved = (self.best_valid_score is None) or (loss_step > self.best_valid_score)
		else:
			has_improved = (self.best_valid_score is None) or (loss_step < self.best_valid_score)

		if has_improved:
			logging.info('Serializing state dicts...')
			torch.save({
					'update': self.restored_run_final_it + it,
					'model': self.model.state_dict(),
					'model_cfg': self.cfg.model,
					'opt': self.opt.state_dict(),
					'loss_train': loss_step,
				}, os.path.join(hydra.utils.get_original_cwd(), f"../checkpoints/{self.cfg.start_timestamp}_{self.cfg.name}.pth"))
			logging.info('Done.')


class EarlyStopping:
	"""This script comes from the repository https://github.com/Bjarten/early-stopping-pytorch/
	Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, goal_maximize=False, model_cfg=None, restored_run_final_it=0):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
			path (str): Path for the checkpoint to be saved to.
							Default: 'checkpoint.pt'
			trace_func (function): trace print function.
							Default: print
			goal_maximize (bool): True if using (ood) accuracy as validation metric for early stop.            
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.early_stop = False
		self.val_metric_best = np.Inf
		self.delta = delta
		self.path = path
		self.trace_func = trace_func
		self.goal_maximize = goal_maximize
		fill_value_deque = 0 if goal_maximize else 100
		self.past_metrics = deque(maxlen=50)
		self.model_cfg = model_cfg
		self.restored_run_final_it = restored_run_final_it

	def __call__(self, val_metric, model, opt, it):
		if self.val_metric_best == np.Inf:
			self.save_checkpoint(val_metric, model, opt, it)
		elif self.metric_has_pushed_average(val_metric):
			self.counter = 0
			if self.metric_is_best(val_metric):
				self.save_checkpoint(val_metric, model, opt, it)
		else:
			self.counter += 1
			self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}. {val_metric} did not push avg {np.mean(self.past_metrics)}.')
			if self.counter >= self.patience:
				self.early_stop = True

		self.past_metrics.append(val_metric)

	def metric_has_pushed_average(self, val_metric):
		if self.goal_maximize:
			if np.mean(self.past_metrics) - val_metric < self.delta:
				return True
			else:
				return False
		else:
			if np.mean(self.past_metrics) - val_metric > self.delta:
				return True
			else:
				return False

	def metric_is_best(self, val_metric):
		if self.goal_maximize:
			return self.val_metric_best < val_metric
		else:
			return self.val_metric_best > val_metric

	def save_checkpoint(self, val_metric, model, opt, it):
		"""Saves model when validation metric improves."""
		if self.verbose:
			self.trace_func(f'Validation metric improved ({self.val_metric_best:.6f} --> {val_metric:.6f}).  Saving model ...')
		torch.save({
				'update': self.restored_run_final_it + it,
				'model': model.state_dict(),
				'model_cfg': self.model_cfg,
				'opt': opt.state_dict(),
				'val_metric': val_metric,
			}, self.path)
		self.val_metric_best = val_metric
