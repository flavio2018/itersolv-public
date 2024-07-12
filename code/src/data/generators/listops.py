import random
import re
import numpy as np
import string
from data.vocabulary import SEP, HAL


MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
FL_SUM_MOD = "[FLSUM"
PROD_MOD = "[PM"
END = "]"
OPERATORS = {
		"i": MIN,
		"a": MAX,
		"e": MED,
		"s": SUM_MOD,
		"f": FIRST,
		"l": LAST,
		"u": FL_SUM_MOD,
		"p": PROD_MOD
	}
OPERATORS_TOK = {
		"i": "MIN",
		"a": "MAX",
		"e": "MED",
		"s": "SM",
		"f": "FIRST",
		"l": "LAST",
		"u": "FLSUM",
		"p": "PM",
	}


def listops_tokens(ops):
	used_operators = [OPERATORS_TOK[op] for op in ops]
	return used_operators + list(string.digits + '[] ' + SEP)    


class ListOpsTree:

	def __init__(self, max_depth, max_args, simplify_last=True, mini_steps=False, ops='iaes', easy=False):
		self.max_depth = max_depth
		self.max_args = max_args
		self.value_p = .25
		self.tree = ()
		self.steps = None
		self.sub_expr = None
		self.values = None
		self.depth = None
		self.simplify_last = simplify_last
		self.mini_steps = mini_steps
		self.easy = easy
		self.OPERATORS = [OPERATORS[op] for op in ops]
		if not self.easy:
			assert self.max_args >= 2, "Cannot have two nesting points with less than 2 operands."
	
	def _generate_tree(self, depth):
		# set probability with which value will be tree or number
		if depth == 1:
			r = 0
		elif depth <= self.max_depth:
			r = random.random()
		else:
			r = 1

		# pick a value at random
		if r > self.value_p:
			value = random.choice(range(10))
			return value

		# choose number of values of expression and recursively 
		# generate values using the same procedure
		else:

			if random.random() > 0.9:
				return random.choice(range(10))

			num_values = random.randint(1, self.max_args)
			
			values = []
			for _ in range(num_values):
				values.append(self._generate_tree(depth + 1))

			# randomly choose op to apply on values
			op = random.choice(self.OPERATORS)
			# build subtree and return to recurse
			t = (op, values[0])
			for value in values[1:]:
				t = (t, value)
			t = (t, END)
		return t
	
	def _generate_depth_k_tree(self, k):
		assert k <= self.max_depth, f'Depth can be at most {self.max_depth}'
		
		if k == 0:
			return random.choice(range(10))
		
		else:

			if self.max_args == 1:
				assert k == 0, "Cannot build expression with 1 arg and depth > 0"
				return random.choice(range(10))

			op = random.choice(self.OPERATORS)
			num_values = self.max_args
			num_depth_k_branches = 1 if self.easy else 2
			pos_depth_k_branches = set(np.random.choice(range(num_values), num_depth_k_branches, replace=False))

			values = []
			for value_pos in range(num_values):
				if value_pos in pos_depth_k_branches:
					values.append(self._generate_depth_k_tree(k-1))
				else:
					values.append(self._generate_depth_k_tree(0))
			
			t = (op, values[0])
			for value in values[1:]:
				t = (t, value)
			t = (t, END)
			return t
			
	def generate_tree(self, depth=None):
		
		if self.max_args == 1:  # halting case
			depth = 0

		if depth is None:
			self.tree = self._generate_tree(1)
		else:
			self.tree = self._generate_depth_k_tree(depth)
		self.depth = self._compute_depth()
		self.steps, self.sub_expr, self.values = self._compute_steps()
   

	@property
	def solution(self):
		if self.steps:
			return self.steps[-1]
		elif self.tree:
			return self.to_string()
		else:
			return None        
	
	def _compute_depth(self):
		if self.tree:
			tree_string = self.to_string()
			return self._compute_depth_expression_str(tree_string)
		else:
			return None

	def _compute_depth_expression_str(self, tree_string):
		depth, max_depth = 0, 0
		for c in tree_string:
			if c == '[':
				depth += 1
			elif c == ']':
				depth -= 1
			if depth > max_depth:
				max_depth = depth
		return max_depth
	
	def _to_string(self, t, parens=True):
		if isinstance(t, str):
			return t
		elif isinstance(t, int):
			return str(t)
		else:
			if parens:
				return '( ' + self._to_string(t[0], parens=parens) + ' ' + self._to_string(t[1], parens=parens) + ' )'
			else:
				return self._to_string(t[0], parens=parens) + self._to_string(t[1], parens=parens)
	
	def to_string(self):
		return self._to_string(self.tree, parens=False)	

	def _compute_op(self, op, args):
		if op == 'MIN':
			sub_expr_value = min(args)
		elif op == 'MAX':
			sub_expr_value = max(args)
		elif op == 'MED':
			sub_expr_value = int(np.median(args))
		elif op == 'SM':
			sub_expr_value = np.sum(args) % 10
		elif op == 'FIRST':
			sub_expr_value = args[0]
		elif op == 'LAST':
			sub_expr_value = args[-1]
		elif op == 'FLSUM':
			sub_expr_value = sum([args[0], args[-1]]) % 10
		elif op == 'PM':
			sub_expr_value = np.prod(args) % 10
		else:
			print(f"Operation {op} not allowed.")
		return sub_expr_value			

	def simplify_one_lvl(self, sample):
		"""Create a simplified version of the expression substituting
		one of the sub-expressions with its result.

		:param sample: the original expression in string format.
		"""
		simplified_sample = sample
		target = sample
		sub_expr_value = sample
		
		sub_expressions = self._get_leaf_subexpressions(sample)

		if sub_expressions != []:
			if self.simplify_last:
				sub_expression = sub_expressions[-1]
			else:
				sub_expression = sub_expressions[0]

			sub_expression_no_parens = sub_expression[1:-1]
			op = re.findall(r'[A-Z]+', sub_expression_no_parens)[0]
			args = [int(v) for v in sub_expression_no_parens[len(op):]]

			if self.mini_steps and (len(args) > 2):
				target = sub_expression_no_parens[:len(op)+2]
				mini_step_value = self._compute_op(op, args[:2])
				replacement = sub_expression_no_parens[:len(op)] + str(mini_step_value)
				sub_expr_value = replacement
			else:
				target = sub_expression
				sub_expr_value = self._compute_op(op, args)
				replacement = str(sub_expr_value)

			if self.simplify_last:
				simplified_sample = replacement.join(sample.rsplit(target, 1))  # substitute at most 1 instance, starting from EOS
			else:
				simplified_sample = sample.replace(target, replacement, 1)  # substitute at most 1 instance
		return simplified_sample, target, str(sub_expr_value)

	@staticmethod
	def _get_leaf_subexpressions(sample):
		sub_expression_re = re.compile(r'\[[A-Z]+[\d{1}]+\]')
		return sub_expression_re.findall(sample)

	def _compute_steps(self):
		sample = self.to_string()
		steps = [sample]
		values = []
		sub_expr = []
		simplified_sample, sub_expression, sub_expr_value = self.simplify_one_lvl(sample)

		while (sample != simplified_sample):
			steps.append(simplified_sample)
			values.append(sub_expr_value)
			sub_expr.append(sub_expression)
			sample = simplified_sample
			simplified_sample, sub_expression, sub_expr_value = self.simplify_one_lvl(sample)

		if (len(steps) == 1) and (len(sample) == 1) and (str.isdigit(sample)):  # case halting
			values.append(HAL)
		
		else:
			values.append(sample)
		
		sub_expr.append(sample)
		return steps, sub_expr, values

	def get_start_end_subexpression(self):
		as_tokenized = lambda s: (s.replace('MAX', 'A')
								   .replace('MIN', 'I')
								   .replace('MED', 'E')
								   .replace('SM', 'M')
								   .replace('FIRST', 'F')
								   .replace('LAST', 'L')
								   .replace('FLSUM', 'U')
								   .replace('PM', 'P'))
		
		pattern = as_tokenized(self.sub_expr[0].replace('[', '\[').replace(']', '\]'))
		expr = as_tokenized(self.to_string())
		match = re.search(pattern, expr)
		return match.start(), match.end()

	def _get_generic_structure_representation(self, tree_str=None):
		if tree_str is None:
			assert self.tree is not None, "Instantiate the tree first."
			tree_str = self.to_string()
		op_re = re.compile(r'(MIN|MAX|SM)')
		digit_re = re.compile(r'[0-9]')
		tree_str = op_re.sub('', tree_str)
		tree_str = digit_re.sub('0', tree_str)
		return tree_str

	def get_solution_chain_stats(self, tree=None):
		if tree is None:
			assert self.tree is not None, "Instantiate the tree first."
			tree = self
		target_sub_expressions = [self._get_leaf_subexpressions(x)[-1] if not x.isdigit() else '' for x in tree.steps]
		num_digits_in_target_sub_expressions = [sum(c.isdigit() for c in s) for s in target_sub_expressions]
		return [(self._compute_depth_expression_str(x), d, x, y) for x, y, d in zip(tree.steps, tree.sub_expr, num_digits_in_target_sub_expressions)]


class ListOpsExpressionGenerator:

	def __init__(self, mini_steps=False, simplify_last=False, ops='iaes', easy=False):
		self.vocab_chars = listops_tokens(ops)
		# super().__init__(listops_tokens(ops), listops_tokens(ops), device, specials_in_x)
		self.simplify_last = simplify_last
		self.mini_steps = mini_steps
		self.ops = ops
		self.easy = easy
		self.process_ops = lambda s: (s.replace('MAX', 'A')
									   .replace('MIN', 'I')
									   .replace('MED', 'E')
									   .replace('SM', 'M')
									   .replace('FIRST', 'F')
									   .replace('LAST', 'L')
									   .replace('FLSUM', 'U')
									   .replace('PM', 'P'))


	def _tokenize_sample(self, sample: str) -> list:
		"""Override method from AbstractGenerator to customize tokenization."""
		sample = self.process_ops(sample)
		tokenized_sample = [c for c in sample]
		output = []
		
		for token in tokenized_sample:
			
			if token == 'A':
				output.append('MAX')
			
			elif token == 'I':
				output.append('MIN')
			
			elif token == 'E':
				output.append('MED')
			
			elif token == 'M':
				output.append('SM')
			
			elif token == 'F':
				output.append('FIRST')
			
			elif token == 'L':
				output.append('LAST')
			
			elif token == 'U':
				output.append('FLSUM')
			
			elif token == 'P':
				output.append('PM')
			
			else:
				output.append(token)

		return output


	def generate_samples(self, num_samples, nesting, num_operands, task, split='train', exact=False):
		samples = [self.generate_sample(nesting, num_operands, split, exact) for _ in range(num_samples)]
		self.subexpressions_positions = [sample.get_start_end_subexpression() for sample in samples]
		self.samples = samples

		X_simplify_w_value, Y_simplify_w_value = self._build_simplify_w_value(samples)
		X_solve_atomic = self._build_solve_atomic_input(X_simplify_w_value)
		X_solve_atomic_no_par = self._build_solve_atomic_input(X_simplify_w_value, no_par=True)
		
		Y_solve_atomic = self._build_solve_atomic_target(X_solve_atomic, samples)
		Y_solve_atomic_no_par = self._build_solve_atomic_target(X_solve_atomic_no_par, samples)
		Y_solve = self._build_solve_target(samples)
		Y_select = self._build_select_target(samples)

		X_by_task = {
			"simplify_with_value": X_simplify_w_value,
			"solve": X_simplify_w_value,
			'solve_atomic': X_solve_atomic,
			'solve_atomic_no_par': X_solve_atomic_no_par,
			"select": X_simplify_w_value,
		}

		Y_by_task = {
			"simplify_with_value": Y_simplify_w_value,
			"solve": Y_solve,
			'solve_atomic': Y_solve_atomic,
			'solve_atomic_no_par': Y_solve_atomic_no_par,
			"select": Y_select,
		}

		if (nesting == 2) and (num_operands == 3):
			X_select_step = self._build_select_step_input(samples)
			Y_select_step = self._build_select_step_target(samples)
			X_by_task["select_step"] = X_select_step
			Y_by_task["select_step"] = Y_select_step

		inputs = []
		targets = []
		
		for task_name in task:
			if not task_name in Y_by_task:
				assert False, f"Wrong task name: {task_name}."
			else:
				assert task_name in ['solve', 'select', 'select_step', 'solve_atomic', 'solve_atomic_no_par']
				inputs.append(X_by_task[task_name])
				targets.append(Y_by_task[task_name])

		if len(targets) == 1:
			inputs = inputs[0]
			targets = targets[0]
		
		return inputs, targets


	def generate_sample(self, max_depth, max_args, split, exact):
		if split is None:
			return self._generate_sample_no_split(max_depth, max_args, exact)
		else:
			return self._generate_sample_in_split(max_depth, max_args, split, exact)


	def _generate_sample_in_split(self, max_depth, max_args, split, exact):
		current_split = ''

		while current_split != split:
			tree = self._generate_sample_no_split(max_depth, max_args, exact)
			sample_hash = hash(tree.to_string())

			if sample_hash % 3 == 0:
				current_split = 'train'

			elif sample_hash % 3 == 1:
				current_split = 'valid'

			else:
				current_split = 'test'

		return tree


	def _generate_sample_no_split(self, max_depth, max_args, exact):
		tree = ListOpsTree(
			max_depth=max_depth,
			max_args=max_args,
			simplify_last=self.simplify_last,
			mini_steps=self.mini_steps,
			ops=self.ops,
			easy=True if (max_args == 1) else self.easy)

		if exact:
			tree.generate_tree(max_depth)

		else:
			tree.generate_tree()

		return tree


	def _build_simplify_w_value(self, samples):
		X_str = []
		Y_str = []

		for sample in samples:
			X_str.append(sample.steps[0])

			if sample.values[0] == HAL:
				Y_str.append(f"{HAL}")
			
			else:   
				Y_str.append(f"{sample.sub_expr[0]}{SEP}{sample.values[0]}")
		
		return X_str, Y_str


	def _build_select_step_input(self, samples):
		return [sample.steps[1] for sample in samples]


	def _build_select_step_target(self, samples):
		return [sample.sub_expr[1] for sample in samples]
	

	def _build_select_target(self, samples):
		Y_str = []

		for sample in samples:
			
			if sample.values[0] == HAL:
				Y_str.append(sample.steps[-1])
			else:   
				Y_str.append(f"{sample.sub_expr[0]}")
		
		return Y_str

	def _build_solve_target(self, samples):
		return [sample.steps[-1] for sample in samples]

	def _build_solve_atomic_input(self, inputs, no_par=False):
		atomic_inputs = []

		for input in inputs:
		
			if (len(input) > 1) and no_par:
				atomic_inputs.append(input[1:-1])	# version w/out parentheses
			else:
				atomic_inputs.append(input)

		return atomic_inputs
		
	def _build_solve_atomic_target(self, X_str, samples):
		Y_str = []

		for x_str, sample in zip(X_str, samples):
			
			if sample.values[0] == HAL:
				Y_str.append(f"{HAL}")
			elif x_str.count('[') == 0:
				operator = ''.join(c for c in x_str if not c.isdigit())
				Y_str.append(operator + sample.steps[-1])
			else:
				Y_str.append(sample.steps[-1])

		return Y_str

	def _build_combiner_target(self, samples):
		return [sample.steps[1] for sample in samples]

	def _build_simplify_target(self, samples):
		combiner_target = self._build_combiner_target(samples)
		return [f"{sample}" for sample in combiner_target]
