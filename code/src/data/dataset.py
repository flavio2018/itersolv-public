import logging
import torch
import pandas as pd
from data.vocabulary import Vocabulary


class ItersolvDataset(torch.utils.data.IterableDataset):

	def __init__(self, dataset_name, split, train_batch_size, eval_batch_size, device, sos, eos, difficulty_split=None, specials_in_x=False):
		self.dataset_name = dataset_name
		self.train_batch_size = train_batch_size
		self.eval_batch_size = eval_batch_size
		self.split = split
		self.device = device
		self.specials_in_x = specials_in_x
		self.sos = sos
		self.eos = eos
		self.difficulty_split = difficulty_split
		self._build_dataset_df(dataset_name, split)
		self._build_vocabulary()
		self._slice_difficulty_split()

	def __iter__(self):
		return self._generate_dict()

	def __len__(self):
		return len(self.df)

	def _build_dataset_df(self, dataset_name, split):
		logging.info(f"Loading dataset {dataset_name}, split {split}...")
		self.df = pd.read_csv(f'../datasets/{dataset_name}/{split}.csv')
		self.df['X'] = self.df['X'].astype('str')
		self.df['Y'] = self.df['Y'].astype('str')
		
	def _slice_difficulty_split(self):
		if self.difficulty_split is not None:
			logging.info(f"Slicing difficulty split: {self.difficulty_split}")
			nesting, num_operands = self.difficulty_split
			self.df = self.df.loc[(self.df['nesting'] == nesting) & (self.df['num_operands'] == num_operands)]	
		logging.info(f"{len(self.df)} total samples in {self.split} split.")

	def _build_vocabulary(self):
		x_vocab_tokens, y_vocab_tokens = self.get_vocab_tokens()
		tokenizer = 'listops' if 'listops' in self.dataset_name else 'char'
		self.vocabulary = Vocabulary(x_vocab_tokens, y_vocab_tokens, self.device, self.sos, self.eos, self.specials_in_x, tokenizer=tokenizer)

	def get_vocab_tokens(self):
		if 'listops' in self.dataset_name:
			return self._get_vocab_tokens_listops()
		else:
			return self._get_vocabs_chars()

	def _get_vocab_tokens_listops(self):
		x_tokens_sets = self.df['X'].apply(Vocabulary._tokenize_listops).apply(lambda s: set(s))
		y_tokens_sets = self.df['Y'].apply(Vocabulary._tokenize_listops).apply(lambda s: set(s))
		return self._build_vocab_tokens_lists(x_tokens_sets, y_tokens_sets)

	def _get_vocabs_chars(self):
		x_chars_sets = self.df['X'].apply(lambda s: set(s))
		y_chars_sets = self.df['Y'].apply(lambda s: set(s))
		return self._build_vocab_tokens_lists(x_chars_sets, y_chars_sets)

	@staticmethod
	def _build_vocab_tokens_lists(x_tokens_sets, y_tokens_sets):
		x_vocab_tokens = set()
		for token_set in x_tokens_sets:
			x_vocab_tokens |= token_set

		y_vocab_tokens = set()
		for token_set in y_tokens_sets:
			y_vocab_tokens |= token_set

		x_vocab_tokens_list = sorted(list(x_vocab_tokens))
		y_vocab_tokens_list = sorted(list(y_vocab_tokens))

		return x_vocab_tokens_list, y_vocab_tokens_list

	@property
	def batch_size(self):
		if self.split == 'train':
			return self.train_batch_size
		else:
			return self.eval_batch_size

	def _generate_dict(self):
		def _continue():
			if self.split == 'train':
				return True
			else:
				self.curr_iter += 1
				return self.curr_iter <= len(self.df) // self.eval_batch_size

		self.curr_iter = 0
		
		while _continue():
			batch_df = self.get_batch(self.batch_size)
			X, Y = batch_df['X'].astype(str).tolist(), batch_df['Y'].astype(str).tolist()
			token_X, token_Y = self.vocabulary.str_to_batch(X), self.vocabulary.str_to_batch(Y, x=False)
			yield token_X, token_Y


	def get_batch(self, batch_size):
		if self.split == 'train':
			return self.df.sample(n=batch_size)
		else:
			return self.df[(self.curr_iter-1)*batch_size:self.curr_iter*batch_size]