import logging
import torch


class Finder(torch.nn.Module):

	def __init__(self, vocabulary):
		super(Finder, self).__init__()
		self.vocabulary = vocabulary

	def forward(self, expression, sub_expression):
		# logging.info('Finder forward.')
		expression = self._to_1hot(expression)
		sub_expression_1hot = self._to_1hot(sub_expression)

		# zero-out padding in sub-expression
		sub_expression_1hot[sub_expression == self.vocabulary.get_special_idx('pad')] = 0

		self._define_conv2d(expression.size(0))
		self._prepare_custom_filter(sub_expression_1hot)

		# add filler zeros to expression
		filter_length = self.conv2d.weight.size(2)
		num_filler_zeros = filter_length
		filler_zeros_1hot = torch.zeros((1, 1, expression.size(2)), device=expression.device)
		filler_zeros = filler_zeros_1hot.tile((expression.size(0), num_filler_zeros, 1))
		expression_filler_zeros = torch.cat((expression, filler_zeros), 1)

		conv2d_out = self.conv2d(expression_filler_zeros.unsqueeze(dim=0))
		position = conv2d_out.argmax(dim=2).squeeze(0)  # needed for case bs=1
		score, _ = conv2d_out.max(dim=2)

		return position.flatten(), score.squeeze(0)

	def _to_1hot(self, batch):
		return torch.nn.functional.one_hot(batch, num_classes=len(self.vocabulary.x_vocab)).type(torch.FloatTensor).to(batch.device)

	def _define_conv2d(self, batch_size):
		self.conv2d = torch.nn.Conv2d(
			in_channels=batch_size,
			out_channels=batch_size,
			kernel_size=(1, 100),  # this is just a placeholder
			padding='valid',
			groups=batch_size)

	def _prepare_custom_filter(self, sub_expressions):
		custom_filter = sub_expressions.unsqueeze(1)
		self.conv2d.weight = torch.nn.Parameter(custom_filter)
		self.conv2d.bias = torch.nn.Parameter(torch.tensor([0.0]*len(sub_expressions), device=sub_expressions.device))
