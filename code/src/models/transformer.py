import hydra
import os
import logging
import math
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Dropout, LayerNorm, Linear, MultiheadAttention, Sequential
from data.vocabulary import EOS, PAD, SEP, HAL, SOS
from models.combiner import NeuralCombiner


@dataclass
class TransformerTestOutput:
	logits: torch.Tensor  # (bs, seq_len, vocab_dim)
	proba: torch.Tensor  # (bs, seq_len)
	pad_idx: int
	eos_idx: int

	@property
	def tokens(self):
		return self.logits.argmax(-1)

	@property
	def pad_mask(self):
		return self.tokens == self.pad_idx

	@property
	def eos_mask(self):
		return self.tokens == self.eos_idx

	@property
	def first_eos_mask(self):
		return self.eos_mask.cumsum(1).cumsum(1) == 1


class Transformer(torch.nn.Module):
	
	def __init__(
		self, d_model, ff_mul, num_heads, num_layers_enc, num_layers_dec, vocabulary, label_pe_enc=False, label_pe_dec=False,
		deterministic=True, n_multi=None, temperature=1, max_range_pe=1000,
		diag_mask_width_below=1, diag_mask_width_above=1, average_attn_weights=True, store_attn_weights=False,
		mha_init_gain=1, num_recurrent_steps=1, multi_fwd_threshold=-1, dropout=0.1, device='cuda'):
		super(Transformer, self).__init__()
		self.vocabulary = vocabulary
		self.device = device
		self.encoder_layers = torch.nn.ModuleList([])
		self.decoder_layers = torch.nn.ModuleList([])
		for _ in range(num_layers_enc):
			self.encoder_layers.append(
				Encoder(d_model, ff_mul, num_heads, dropout=dropout, label_pe=label_pe_enc, max_range_pe=max_range_pe, 
					diag_mask_width_below=diag_mask_width_below, diag_mask_width_above=diag_mask_width_above, 
					average_attn_weights=average_attn_weights, store_attn_weights=store_attn_weights,
					device=device))
			self.encoder_layers[-1]._init_mha(mha_init_gain)
		for _ in range(num_layers_dec):
			self.decoder_layers.append(
				Decoder(d_model, ff_mul, num_heads, dropout=dropout, label_pe=label_pe_dec, max_range_pe=max_range_pe,
				 average_attn_weights=average_attn_weights, store_attn_weights=store_attn_weights, device=device))
			self.decoder_layers[-1]._init_mha(mha_init_gain)
		self.d_model = d_model
		self.idx_PAD_x = vocabulary.get_special_idx('pad')
		self.idx_PAD_y = vocabulary.get_special_idx('pad', x=False)
		self.idx_SOS_y = vocabulary.get_special_idx('sos', x=False)
		self.idx_EOS_y = vocabulary.get_special_idx('eos', x=False)
		self.len_x_vocab, self.len_y_vocab = len(vocabulary.x_vocab), len(vocabulary.y_vocab)
		self.x_emb = Embedding(num_embeddings=self.len_x_vocab, embedding_dim=self.d_model, padding_idx=self.idx_PAD_x, device=self.device)
		self.y_emb = Embedding(num_embeddings=self.len_y_vocab, embedding_dim=self.d_model, padding_idx=self.idx_PAD_y, device=self.device)
		self.final_proj = Linear(self.d_model, self.len_y_vocab, device=self.device)
		for enc in self.encoder_layers:
			enc.set_vocabulary(vocabulary)
		self.deterministic = deterministic
		self.store_attn_weights = store_attn_weights
		self.n_multi = n_multi
		self.temperature = temperature
		self.num_recurrent_steps = num_recurrent_steps
		self.multi_fwd_threshold = multi_fwd_threshold

	def load_model_weights(self, ckpt):
		logging.info(f'Loading model weights from checkpoint {ckpt}...')
		torch_ckpt = torch.load(os.path.join(hydra.utils.get_original_cwd(), f'../checkpoints/{ckpt}'), map_location=self.device)
		self.load_state_dict(torch_ckpt['model'])

	def forward(self, X, Y=None, tf=False):
		if self.store_attn_weights:
			for decoder_layer in self.decoder_layers:
				decoder_layer.self_attn = []
				decoder_layer.cross_attn = []

		if self.n_multi is not None:
			assert not self.deterministic
			self.combiner = NeuralCombiner(X.size(0))
			self.combiner.get_vocab_info(self.vocabulary)
			return self._multi_fwd(X)
		elif Y is not None:
			return self._fwd(X, Y, tf=tf)
		else:
			return self._test_fwd(X)

	def _fwd(self, X, Y, tf=False):
		src_mask = (X == self.idx_PAD_x)
		tgt_mask = (Y == self.idx_PAD_y)
		X = self.x_emb(X)

		if not tf:
			X, src_mask = self._encoder(X, src_mask)
			Y_greedy_pred_tokens = Y[:, 0].unsqueeze(1)
			output = F.one_hot(Y_greedy_pred_tokens, num_classes=self.len_y_vocab).type(torch.FloatTensor).to(X.device)  # placeholder
			for t in range(Y.size(1)):
				Y_pred = self.y_emb(Y_greedy_pred_tokens)
				self.Y_logits = self._decoder(X, Y_pred, src_mask, None)
				Y_pred = self.final_proj(self.Y_logits)
				Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
				output = torch.concat([output, Y_pred], dim=1) 
				pred_idx = Y_pred.argmax(-1) 
				Y_greedy_pred_tokens = torch.concat([Y_greedy_pred_tokens, pred_idx], dim=1)
			return output[:, 1:, :]
		else:
			Y = self.y_emb(Y)
			X, src_mask = self._encoder(X, src_mask)
			self.Y_logits = self._decoder(X, Y, src_mask, tgt_mask)
			return self.final_proj(self.Y_logits)
	
	def _encoder(self, X, src_mask):
		for _ in range(self.num_recurrent_steps):
			for encoder_layer in self.encoder_layers:
				X = encoder_layer(X, src_mask)
		return X, src_mask

	def _decoder(self, X, Y, src_mask, tgt_mask):
		for l_idx, decoder_layer in enumerate(self.decoder_layers):
			Y = decoder_layer(X, Y, src_mask, tgt_mask)
		return Y

	def _test_fwd(self, X):
		it, max_it = 0, 100
		src_mask = (X == self.idx_PAD_x)
		X = self.x_emb(X)
		encoding, src_mask = self._encoder(X, src_mask)
		stopped = torch.zeros(X.size(0)).type(torch.BoolTensor).to(X.device)
		output = torch.tile(F.one_hot(torch.tensor([self.idx_SOS_y]), num_classes=self.len_y_vocab), dims=(X.size(0), 1, 1)).type(torch.FloatTensor).to(X.device)
		Y_sampled_pred_tokens = output.argmax(-1)
		output_proba = None
		list_output_proba = []

		while not stopped.all() and (it < max_it):
			it += 1
			Y_pred = self.y_emb(Y_sampled_pred_tokens)
			self.Y_logits = self._decoder(encoding, Y_pred, src_mask, None)
			Y_pred = self.final_proj(self.Y_logits)
			Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
		
			if self.deterministic:
				output = torch.concat([output, Y_pred], dim=1)
				pred_idx = Y_pred.argmax(-1)
			else:
				tokens_distrib = F.softmax(Y_pred.squeeze(1) / self.temperature, dim=-1)  # we can squeeze bc we take 1 sample
				pred_idx = torch.multinomial(tokens_distrib, num_samples=1)
				token_proba = tokens_distrib.gather(dim=-1, index=pred_idx)
				list_output_proba.append(token_proba)
				output_proba = torch.concat(list_output_proba, dim=-1)
				Y_sample = F.one_hot(pred_idx, num_classes=self.len_y_vocab).type(torch.FloatTensor).to(X.device)
				output = torch.concat([output, Y_sample], dim=1)

			Y_sampled_pred_tokens = output.argmax(-1)
			stopped = torch.logical_or((pred_idx.squeeze() == self.idx_EOS_y), stopped)
		
		return TransformerTestOutput(logits=output[:, 1:, :], proba=output_proba, pad_idx=self.idx_PAD_y, eos_idx=self.idx_EOS_y)

	def _update_outputs_log_proba(self, outputs_log_proba, X, argmax_solver_out, multi_idx, halted):
		"""Compute avg proba associated to output tokens excluding those after first EOS."""
			
		if halted.all():
			outputs_log_proba[halted, multi_idx] = torch.zeros_like(outputs_log_proba[halted, 0])
		
		else:
			mask_first_eos_tok = ((argmax_solver_out == self.idx_EOS_y).cumsum(1).cumsum(1) == 1)				
			# cover case no EOS
			exist_eos_in_seq = mask_first_eos_tok.any(-1)
			where_first_eos = torch.argwhere(mask_first_eos_tok)[:, 1]
			where_first_eos_fixed = torch.zeros(X.size(0)).to(where_first_eos)
			where_first_eos_fixed[exist_eos_in_seq] = where_first_eos
			where_first_eos_fixed[~exist_eos_in_seq] = 2000  # some very big value
			
			positions_batch = torch.ones_like(argmax_solver_out).cumsum(1) - 1
			positions_after_first_eos = positions_batch > where_first_eos_fixed.unsqueeze(1)
			self.output_proba_last_batch[positions_after_first_eos] = 1
			# outputs_lengths = torch.zeros(X.size(0)).to(where_first_eos)
			# outputs_lengths[exist_eos_in_seq] = where_first_eos + 1
			output_log_proba_last_batch = torch.log(self.output_proba_last_batch + 1e-10)

			outputs_log_proba[~halted, multi_idx] = output_log_proba_last_batch[~halted].sum(-1)  # / outputs_lengths[~halted]
			outputs_log_proba[~halted, multi_idx] = torch.nan_to_num(outputs_log_proba[~halted, multi_idx], nan=0, posinf=0, neginf=0)
		
		logging.info("outputs_log_proba")
		logging.info(outputs_log_proba[:, multi_idx])
			
		return outputs_log_proba

	def _update_finder_scores(self, finder_scores, X, solver_out, multi_idx, halted):
		splittable = self.combiner._is_splittable(solver_out)
		splittable_or_halted = torch.bitwise_or(~splittable, halted)

		if halted.all():
			finder_scores[halted, multi_idx] = torch.ones_like(finder_scores[halted, 0])

		elif splittable_or_halted.all():  # any sequence is either halting or non-splittable
			finder_scores[halted, multi_idx] = torch.ones_like(finder_scores[halted, 0])
			finder_scores[~splittable, multi_idx] = torch.zeros_like(finder_scores[~splittable, 0])
		
		else:
			splitted_expressions = torch.zeros_like(solver_out)
			expression, _, _, _ = self.combiner._split(solver_out[splittable])
		
			expressions_lengths = torch.zeros(X.size(0), device=X.device).long()
			mask_sep_tok = ((solver_out[splittable].argmax(-1) == self.combiner.sep_tok_idx).cumsum(1).cumsum(1) == 1)
			mask_pad_tok = ((solver_out[~splittable].argmax(-1) == self.idx_EOS_y).cumsum(1).cumsum(1) == 1)
			expressions_lengths[splittable] = torch.argwhere(mask_sep_tok)[:, 1]
			expressions_lengths[~splittable] = 0

			# prevent padded values from matching in the filter
			expression_mask = (expression == self.combiner.pad_tok_idx)
			expression_mask_3d = expression_mask.unsqueeze(-1).tile(1, 1, self.combiner.len_y_vocab)
			expression_1hot = self.combiner._to_1hot(expression)
			expression_1hot[expression_mask_3d] = 0
			
			splitted_expressions[splittable, :expression_1hot.size(1), :] = expression_1hot
			splitted_expressions[~splittable, :, self.idx_SOS_y] = 1
			finder_scores[~halted, multi_idx] = self.combiner.finder.get_expressions_match(X[~halted], splitted_expressions[~halted, :expressions_lengths[splittable].max()])
		
		logging.info("finder_scores")
		logging.info(finder_scores[:, multi_idx])

		if (not halted.all()) and (not splittable_or_halted.all()):
			finder_scores[~halted, multi_idx] = finder_scores[~halted, multi_idx] / expressions_lengths[~halted]
			finder_scores[~halted, multi_idx] = torch.nan_to_num(finder_scores[~halted, multi_idx], nan=0, posinf=0, neginf=0)
			logging.info("expressions_lengths")
			logging.info(expressions_lengths)
		
		logging.info("normalized finder_scores")
		logging.info(finder_scores[:, multi_idx])
			
		return finder_scores

	def _multi_fwd(self, X):

		def _cut_at_first_dot(x_str):
			pos_eos = [pos for pos, char in enumerate(x_str) if char == EOS]
			if len(pos_eos) > 0:
				return x_str[:pos_eos[0]]
			else:
				return x_str

		outputs_log_proba = torch.zeros(X.size(0), self.n_multi, device=X.device)
		multi_outputs = []
		finder_scores = torch.zeros_like(outputs_log_proba)
		
		for multi_idx in range(self.n_multi):
			solver_out = self._test_fwd(X)
			multi_outputs.append(solver_out)

			argmax_solver_out = solver_out.argmax(-1)
			halted = argmax_solver_out[:, 0] == torch.tensor(self.vocabulary.y_vocab[HAL], device=X.device)

			logging.info("multi_outputs[-1]")
			logging.info(self.vocabulary.batch_to_str(multi_outputs[-1]))
			
			outputs_log_proba = self._update_outputs_log_proba(outputs_log_proba, X, argmax_solver_out, multi_idx, halted)
			finder_scores = self._update_finder_scores(finder_scores, X, solver_out, multi_idx, halted)

		final_outputs = []
		multi_outputs_str = [self.vocabulary.batch_to_str(o, x=False) for o in multi_outputs]
		multi_outputs_T = [list(i) for i in zip(*multi_outputs_str)]
		multi_outputs_T = [[_cut_at_first_dot(o) for o in outputs] for outputs in multi_outputs_T]
		
		for input_idx, outputs_per_input in enumerate(multi_outputs_T):
			avg_outputs_log_probas = defaultdict(list)
			outputs_with_finder_score_1 = set()
			
			for output_idx, output in enumerate(outputs_per_input):
				output = _cut_at_first_dot(output)
				
				if (finder_scores[input_idx, output_idx] == 1) or (output == HAL):
					avg_outputs_log_probas[output].append(outputs_log_proba[input_idx, output_idx].item())
					outputs_with_finder_score_1.add(output)
			
			avg_outputs_log_probas = {k: np.array(v).mean() for k,v in avg_outputs_log_probas.items()}
			avg_outputs_log_probas = {k: v for k, v in sorted(avg_outputs_log_probas.items(), key=lambda item: item[1], reverse=True)}
			counter_valid_outputs = Counter([output for output in outputs_per_input if output in outputs_with_finder_score_1])
			logging.info('')
			logging.info(avg_outputs_log_probas)
			logging.info(outputs_with_finder_score_1)
			logging.info(str(counter_valid_outputs))

			if len(outputs_with_finder_score_1) == 0:
				final_outputs.append(PAD)
			else:
				output_with_highest_avg_log_proba = max(avg_outputs_log_probas, key=avg_outputs_log_probas.get)
				
				if (output_with_highest_avg_log_proba == HAL) and (len(outputs_with_finder_score_1) > 1):
					avg_outputs_log_probas.pop(output_with_highest_avg_log_proba)
					final_outputs.append(max(avg_outputs_log_probas, key=avg_outputs_log_probas.get) + EOS)
				elif avg_outputs_log_probas[output_with_highest_avg_log_proba] > self.multi_fwd_threshold:
					final_outputs.append(output_with_highest_avg_log_proba + EOS)
				else:
					final_outputs.append(PAD)

		final_outputs_batch = self.vocabulary.str_to_batch(final_outputs, x=False)
		return final_outputs_batch

	def _test_fwd_encode_step(self, X):
		src_mask = (X == self.idx_PAD_x)
		X = self.x_emb(X)
		encoding, src_mask = self._encoder(X, src_mask)
		return encoding, src_mask

	def _test_fwd_decode_step(self, encoding, src_mask, Y_pred_v):
		Y_pred = self.y_emb(Y_pred_v)
		self.Y_logits = self._decoder(encoding, Y_pred, src_mask, None)
		Y_pred = self.final_proj(self.Y_logits)
		Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
		return Y_pred


class Encoder(torch.nn.Module):

	def __init__(self, d_model, ff_mul, num_heads, dropout=0.1, label_pe=False, max_range_pe=1000, 
	 diag_mask_width_below=1, diag_mask_width_above=1, average_attn_weights=True, 
	 store_attn_weights=False, device='cpu'):
		super(Encoder, self).__init__()
		self.device = device
		positional_encoding = _gen_timing_signal(max_range_pe, d_model)
		self.register_buffer('positional_encoding', positional_encoding)
		
		self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
		self.dropout1 = Dropout(dropout)
		self.layer_norm1 = LayerNorm(d_model)
		self.transition_fn = Sequential(Linear(d_model, ff_mul*d_model),
										torch.nn.ReLU(),
										Linear(ff_mul*d_model, d_model))
		self.dropout2 = Dropout(dropout)
		self.layer_norm2 = LayerNorm(d_model)
		self.label_pe = label_pe
		self.diag_mask_width_below = diag_mask_width_below
		self.diag_mask_width_above = diag_mask_width_above
		self.average_attn_weights = average_attn_weights
		self.store_attn_weights = store_attn_weights

	def _init_mha(self, gain=1):
	    if self.MHSA._qkv_same_embed_dim:
	        torch.nn.init.xavier_uniform_(self.MHSA.in_proj_weight, gain=gain)
	    else:
	        torch.nn.init.xavier_uniform_(self.MHSA.q_proj_weight, gain=gain)
	        torch.nn.init.xavier_uniform_(self.MHSA.k_proj_weight, gain=gain)
	        torch.nn.init.xavier_uniform_(self.MHSA.v_proj_weight, gain=gain)

	def set_vocabulary(self, vocabulary):
		self.vocabulary = vocabulary

	def forward(self, X, src_mask):
		X = self._pe(X)
		X = self._encoder(X, src_mask)
		return X

	def _encoder(self, X, src_mask):
		if self.diag_mask_width_below is not None:
			assert self.diag_mask_width_above is not None
			attn_mask = _gen_mask_window_around_diag(src_mask=src_mask, num_heads=self.MHSA.num_heads,
			 below=self.diag_mask_width_below, above=self.diag_mask_width_above).to(X.device)
		else:
			attn_mask = None

		if attn_mask is not None:
			attn_mask = _adapt_mask_to_padding(attn_mask, src_mask, self.MHSA.num_heads)
			self.attn_mask_padfix = attn_mask

		Xt, self_attn = self.MHSA(X, X, X, attn_mask=attn_mask, key_padding_mask=src_mask, average_attn_weights=self.average_attn_weights)
		if self.store_attn_weights:
			self.self_attn = self_attn
		X = X + self.dropout1(Xt)
		X = self.layer_norm1(X)
		X = X + self.dropout2(self.transition_fn(X))
		X = self.layer_norm2(X)
		return X

	def _pe(self, X):
		if self.label_pe:
			max_seq_len = X.size(1)
			max_pe_pos = self.positional_encoding.size(1)
			self.label_pe_val, idx = torch.sort(torch.randint(low=0, high=max_pe_pos, size=(max_seq_len,)))
			return X + self.dropout1(self.positional_encoding[:, self.label_pe_val, :])
		else:
			return X + self.dropout1(self.positional_encoding[:, :X.size(1), :])

	def _get_raw_label_pe(self):
		return self.dropout1(self.positional_encoding[:, self.label_pe_val, :])


class Decoder(torch.nn.Module):
	
	def __init__(self, d_model, ff_mul, num_heads, dropout=0.1, label_pe=False, max_range_pe=1000,
	 average_attn_weights=True, store_attn_weights=False, device='cpu'):
		super(Decoder, self).__init__()
		self.device = device
		positional_encoding = _gen_timing_signal(max_range_pe, d_model)
		self.register_buffer('positional_encoding', positional_encoding)
		
		self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
		self.dropout1 = Dropout(dropout)
		self.layer_norm1 = LayerNorm(d_model)
		self.MHA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
		self.dropout2 = Dropout(dropout)
		self.layer_norm2 = LayerNorm(d_model)
		self.transition_fn = Sequential(Linear(d_model, ff_mul*d_model),
										torch.nn.ReLU(),
										Linear(ff_mul*d_model, d_model))
		self.dropout3 = Dropout(dropout)
		self.layer_norm3 = LayerNorm(d_model)
		self.device = device
		self.label_pe = label_pe
		self.average_attn_weights = average_attn_weights
		self.store_attn_weights = store_attn_weights
	
	def _init_mha(self, gain=1):
	    for mha in [self.MHSA, self.MHA]:
		    if mha._qkv_same_embed_dim:
		        torch.nn.init.xavier_uniform_(mha.in_proj_weight, gain=gain)
		    else:
		        torch.nn.init.xavier_uniform_(mha.q_proj_weight, gain=gain)
		        torch.nn.init.xavier_uniform_(mha.k_proj_weight, gain=gain)
		        torch.nn.init.xavier_uniform_(mha.v_proj_weight, gain=gain)

	def forward(self, X, Y, src_mask, tgt_mask):
		Y = self._pe(Y)
		if self.label_pe:
			self.pe_Y = self._get_raw_label_pe()
			self._pe(X)
			self.pe_X = self._get_raw_label_pe()
		Y = self._decoder(X, Y, src_mask, tgt_mask)
		return Y

	def _decoder(self, X, Y, src_mask, tgt_mask):
		Yt, self_attn = self.MHSA(Y, Y, Y, attn_mask=_gen_bias_mask(Y.size(1), self.device),
		 key_padding_mask=tgt_mask, average_attn_weights=self.average_attn_weights)
		Y = Y + self.dropout1(Yt)
		Y = self.layer_norm1(Y)
		
		Yt, cross_attn = self.MHA(Y, X, X, attn_mask=None,
		 key_padding_mask=src_mask, average_attn_weights=self.average_attn_weights)
		if self.store_attn_weights:
			self.self_attn.append(self_attn)
			self.cross_attn.append(cross_attn)
		Y = Y + self.dropout2(Yt)
		Y = self.layer_norm2(Y)
		Y = self.dropout3(self.transition_fn(Y))
		Y = self.layer_norm3(Y)
		return Y

	def _pe(self, X):
		if self.label_pe:
			max_seq_len = X.size(1)
			max_pe_pos = self.positional_encoding.size(1)
			self.label_pe_val, idx = torch.sort(torch.randint(low=0, high=max_pe_pos, size=(max_seq_len,)))
			return X + self.dropout1(self.positional_encoding[:, self.label_pe_val, :])
		else:
			return X + self.dropout1(self.positional_encoding[:, :X.size(1), :])

	def _get_raw_label_pe(self):
		return self.dropout1(self.positional_encoding[:, self.label_pe_val, :])


def _gen_bias_mask(max_len, device):
	"""
	Generates bias values (True) to mask future timesteps during attention
	"""
	np_mask = np.triu(np.full([max_len, max_len], 1), 1)
	torch_mask = torch.from_numpy(np_mask).type(torch.BoolTensor).to(device)
	
	return torch_mask


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
	"""
	Generates a [1, length, channels] timing signal consisting of sinusoids
	Adapted from:
	https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
	"""
	position = np.arange(length)
	num_timescales = channels // 2
	log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
	inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(float) * -log_timescale_increment)
	scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

	signal = np.zeros((scaled_time.shape[0], 2*scaled_time.shape[1]))
	signal[:, 0::2] = np.sin(scaled_time)
	signal[:, 1::2] = np.cos(scaled_time)
	signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
					'constant', constant_values=[0.0, 0.0])
	signal =  signal.reshape([1, length, channels])

	return torch.from_numpy(signal).type(torch.FloatTensor)


def _gen_mask_window_around_diag(src_mask, num_heads, below: int = 1, above: int = 1):
	"""Generates a mask around the diagonal of the self attention.
	
	:param src_mask: padding mask of the source sequence.
	:param num_heads: number of heads of the model.
	:param below: integer representing the width of the mask below the diagonal.
	:param above: integer representing the width of the mask above the diagonal.
	"""
	seq_lens = (~src_mask).sum(1)
	max_seq_len = src_mask.size(1)
	bs = len(seq_lens)
	mask = torch.ones(bs*num_heads, max_seq_len, max_seq_len)
	
	# this prevents errors when creating the mask
	# in the case of seqs shorter than window_size
	if below > max_seq_len:
		below = max_seq_len
	if above > max_seq_len:
		above = max_seq_len
	
	for w in range(-below, above+1, 1):
		mask -= torch.diag(torch.ones(max_seq_len-abs(w)), w)

	return mask.bool()


def _adapt_mask_to_padding(mask, src_mask, num_heads):
	seq_lens = (~src_mask).sum(1)
	
	for seq_idx, seq_len in enumerate(seq_lens):
		mask[num_heads*seq_idx:seq_idx*num_heads+num_heads, seq_len:, :] = False
		mask[num_heads*seq_idx:seq_idx*num_heads+num_heads, :, seq_len:] = False

	return mask
