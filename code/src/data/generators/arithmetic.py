import copy
import string
import numpy as np
from data.vocabulary import SEP, HAL


class ArithmeticExpression:
	
	def __init__(self, min_operand_value=-99, max_operand_value=99, modulo=100, operators='+-*', mini_steps=False, simplify_signs=False, easy=False, lev_nes_2=0):
		self.min_operand_value = min_operand_value
		self.max_operand_value = max_operand_value
		self.operators = operators
		self.steps = []
		self.modulo = modulo
		self.mini_steps = mini_steps
		self.simplify_signs = simplify_signs
		self.easy = easy
		self.lev_nes_2 = lev_nes_2

	def _make_structure(self, current_nesting):

		if self.exact:

			if self.max_num_operands == 1:  # case single number, halting condition
				assert self.input_nesting == 1, "Cannot request single op expression of nesting > 1."
				return [self.input_nesting]

			num_operands = 2 if np.random.rand() > .66 else self.max_num_operands  # enable multiplication
			num_nesting_pts = num_nesting_pts = 1 if self.easy else 2
			
			if (current_nesting > self.lev_nes_2):
				num_nesting_pts = 1

			depth_nesting_pts = [current_nesting]*num_nesting_pts
		
		else:

			# case single number, halting condition
			nesting_lvl = np.random.randint(1, current_nesting+1)

			if (current_nesting == self.input_nesting) and (nesting_lvl == 1):
				num_operands = np.random.randint(1, self.max_num_operands+1)
				
				if num_operands == 1:
					return [current_nesting]

			num_operands = np.random.randint(2, self.max_num_operands+1)
			num_nesting_pts = np.random.randint(1, num_operands+1)
			depth_nesting_pts = np.random.randint(1, current_nesting+1, num_nesting_pts)
		
		nesting_pts = list(set(np.random.choice(range(num_operands), num_nesting_pts, replace=False)))
		nesting_pts_depth = { p: d for (p, d) in zip(nesting_pts, depth_nesting_pts) }
		
		structure = []
		
		for pos in range(num_operands):
			
			if pos in nesting_pts_depth:
				
				if nesting_pts_depth[pos] == 1:
					structure.append(current_nesting)

				else:
					structure.append(self._make_structure(current_nesting-1))
			
			else:
				structure.append(current_nesting)
				
		return structure
	
	def _add_operands_and_ops_placeholders(self, structure):
		operands = []
		
		for placeholder in structure:
			
			if isinstance(placeholder, list):
				operands.append(self._add_operands_and_ops_placeholders(placeholder))
				operands.append('?')
			
			else:
				operands.append(np.random.randint(self.min_operand_value, self.max_operand_value+1))
				operands.append('?')
		
		return operands[:-1]
	
	def _add_operators(self, expression_ops_placeholders):      
		expression = []
		operators_wout_prod = self.operators.replace('*', '')
		expr_has_more_two_ops = len(expression_ops_placeholders) > 3
		
		for pos, operand in enumerate(expression_ops_placeholders):
			
			if isinstance(operand, list):
				expression.append(self._add_operators(operand))
			
			elif operand == '?':
				
				if expr_has_more_two_ops:
					expression.append(operators_wout_prod[np.random.randint(len(operators_wout_prod))])
				
				else:

					if self.exact and (self.max_num_operands > 2):
						expression.append('*')
					else:
						if np.random.rand() > .66:
							expression.append('*')
						else:
							expression.append(operators_wout_prod[np.random.randint(len(operators_wout_prod))])

			else:
				expression.append(operand)
		
		return expression

	@staticmethod
	def _simplify_signs(string_expr):
		return (string_expr.replace('--', '+')
						   .replace('+-', '-')
						   .replace('-+', '-'))

	def build(self, nesting, num_operands, exact=False):
		self.max_num_operands = num_operands
		self.input_nesting = nesting
		self.exact = exact
		self.current_depth = 0
		
		if (not self.easy) and exact:
			assert self.max_num_operands >= 2, "Cannot have two nesting points with less than 2 operands."
		
		structure = self._make_structure(nesting)
		expression_ops_placeholders = self._add_operands_and_ops_placeholders(structure)
		self.expression = self._add_operators(expression_ops_placeholders)
		self._compute_steps()
	
	def _compute_steps(self):
		self.steps = []
		expression = copy.deepcopy(self.expression)
		
		reduced_expression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(expression)
		self.steps.append((copy.deepcopy(reduced_expression), subexpression_string, subexpression_value))
		
		while isinstance(reduced_expression, list):
			reduced_expression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(reduced_expression)
			self.steps.append((copy.deepcopy(reduced_expression), subexpression_string, subexpression_value))
		
	def _compute_rightmost_deepmost_step(self, expression=None):

		if expression is None:
			assert self.expression is not None, "Cannot evaluate before building expression"
			expression = copy.deepcopy(self.expression)
		
		expression_types = set([type(v) for v in expression])
		
		if list in expression_types:
			
			for value_pos in range(len(expression)-1, -1, -1):
				value = expression[value_pos]
				
				if isinstance(value, list):
					new_subexpression, subexpression_string, subexpression_value = self._compute_rightmost_deepmost_step(value)
					reduced_expression = expression
					reduced_expression[value_pos] = new_subexpression
					
					return (reduced_expression, subexpression_string, subexpression_value)
		
		else:

			if self.mini_steps:
				expression_string = ''.join(str(v) for v in expression[:3])
				value = eval(expression_string)
				value_modulo = 0 if value == 0 else value % (np.sign(value) * self.modulo)
				
				if len(expression) == 1:   # case halting condition

					if value_modulo == expression[0]:
						reduced_expression = HAL
						value_modulo = HAL

						if self.simplify_signs:
							expression_string = self._simplify_signs(expression_string)

						return (reduced_expression, expression_string, value_modulo)
					
				if len(expression) > 3:
					reduced_expression = [value_modulo] + expression[3:]
					
					if self.simplify_signs:
						expression_string = self._simplify_signs(expression_string)
					
					return (reduced_expression, expression_string, value_modulo)
				
				else:
					reduced_expression = value_modulo

					if self.simplify_signs:
						expression_string = self._simplify_signs(expression_string)            
					
					return (reduced_expression, f"({expression_string})", value_modulo)
			
			else:
				expression_string = ''.join(str(v) for v in expression)
				value = eval(expression_string)
				value_modulo = 0 if value == 0 else value % (np.sign(value) * self.modulo)

				if self.simplify_signs:
					expression_string = self._simplify_signs(expression_string)
				
				return (value_modulo, f"({expression_string})", value_modulo)
	
	def to_string(self, expression=None):
		string_expr = ''

		if expression is None:
			expression = self.expression
		
		if isinstance(expression, list):

			if len(expression) == 1:
				return str(expression[0])
			
			for value in expression:
				
				if isinstance(value, list):
					string_expr += self.to_string(value)
				
				else:
					string_expr += str(value)

		elif isinstance(expression, str) and expression == '$':
			return expression

		else:
			string_expr = str(expression)
			return string_expr

		if self.simplify_signs:
			string_expr = self._simplify_signs(string_expr)
		
		return f"({string_expr})"
	
	def __repr__(self):
		return self.to_string()

	def get_solution_chain_stats(self):
		return [(self._compute_depth_expression_str(x), 2, x, y) for x, y in zip(self.solution_chain, self.sub_expr)]

	@property
	def solution_chain(self):
		if self.steps is None:
			return None
		else:
			return [self.to_string()] + [self.to_string(step[0]) for step in self.steps]

	@property
	def sub_expr(self):
		if self.steps is None:
			return None
		else:
			return [step[1] for step in self.steps] + [str(self.steps[-1][0])]

	def _compute_depth_expression_str(self, expression_string):
		depth, max_depth = 0, 0
		for c in expression_string:
			if c == '(':
				depth += 1
			elif c == ')':
				depth -= 1
			if depth > max_depth:
				max_depth = depth
		return max_depth
	

class ArithmeticExpressionGenerator:
	
	def __init__(self,
				 min_operand_value=-99,
				 max_operand_value=99,
				 modulo=100,
				 operators='+-*',
				 mini_steps=False,
				 easy=False):     
		self.vocab_chars = string.digits + '()+*-' + SEP
		# super().__init__(vocab_chars, vocab_chars, device, specials_in_x)
		self.min_operand_value = min_operand_value
		self.max_operand_value = max_operand_value
		self.modulo = modulo
		self.operators = operators
		self.mini_steps = mini_steps
		self.easy = easy
			
	def generate_samples(self, num_samples, nesting, num_operands, task, split='train', exact=False):
		samples = [self.generate_sample(nesting, num_operands, split, exact) for _ in range(num_samples)]
		self.samples = samples

		X_simplify_w_value, Y_simplify_w_value = self._build_simplify_w_value(samples)
		Y_combiner = self._build_combiner_target(samples)
		Y_solve = self._build_solve_target(samples)
		Y_solve_atomic = self._build_solve_atomic_target(samples)
		Y_simplify = self._build_simplify_target(samples)
		Y_select = self._build_select_target(samples)

		X_by_task = {
			'simplify_with_value': X_simplify_w_value,
			'combiner': X_simplify_w_value,
			'solve': X_simplify_w_value,
			'solve_atomic': X_simplify_w_value,
			'simplify': X_simplify_w_value,
			'select': X_simplify_w_value,
		}
		
		Y_by_task = {
			'simplify_with_value': Y_simplify_w_value,
			'combiner': (Y_simplify_w_value, Y_combiner),
			'solve': Y_solve,
			'solve_atomic': Y_solve_atomic,
			'simplify': Y_simplify,
			'select': Y_select,
		}

		if (nesting == 2) and (num_operands == 2):
			X_by_task['select_s1'] = self._build_select_step_input(samples, 's1')
			Y_by_task['select_s1'] = self._build_select_step_target(samples, 's1')

		if (nesting == 3) and (num_operands == 2):
			for step in ['s1', 's2', 's3', 's4']:
				X_by_task[f'select_{step}'] = self._build_select_step_input(samples, step)
				Y_by_task[f'select_{step}'] = self._build_select_step_target(samples, step)

		inputs = []
		targets = []
		
		for task_name in task:
			if not task_name in Y_by_task:
				assert False, f"Wrong task name: {task_name}."
			else:
				inputs.append(X_by_task[task_name])
				targets.append(Y_by_task[task_name])

		if len(targets) == 1:
			inputs = inputs[0]
			targets = targets[0]
		
		return inputs, targets
	
	def generate_sample(self, nesting, num_operands, split, exact):
		if split is None:
			return self._generate_sample_no_split(nesting, num_operands, exact)
		else:
			return self._generate_sample_in_split(nesting, num_operands, split, exact)

	def _generate_sample_in_split(self, nesting, num_operands, split, exact):
		current_split = ''
		
		while current_split != split:
			expression = self._generate_sample_no_split(nesting, num_operands, exact)
			sample_hash = hash(expression.to_string())

			if sample_hash % 3 == 0:
				current_split = 'train'
			
			elif sample_hash % 3 == 1:
				current_split = 'valid'
			
			else:
				current_split = 'test'
		return expression

	def _generate_sample_no_split(self, nesting, num_operands, exact):
		simplify_signs = np.random.rand() > .2
		expression = ArithmeticExpression(min_operand_value=self.min_operand_value,
										  max_operand_value=self.max_operand_value,
										  modulo=self.modulo,
										  operators=self.operators,
										  mini_steps=self.mini_steps,
										  simplify_signs=simplify_signs,
										  easy=True if (num_operands == 1) else self.easy)
		expression.build(nesting=nesting, num_operands=num_operands, exact=exact)
		return expression
	
	def _build_simplify_w_value(self, samples):
		X_str = []
		Y_str = []

		for sample in samples:
			X_str.append(sample.to_string())

			if '+-' in X_str[-1]:
				Y_str.append(f"+-{SEP}-")
			
			elif '-+' in X_str[-1]:
				Y_str.append(f"-+{SEP}-")
			
			elif '--' in X_str[-1]:
				Y_str.append(f"--{SEP}+")

			elif sample.steps[0][2] == HAL:
				Y_str.append(f"{HAL}")
			
			else:
				Y_str.append(f"{sample.steps[0][1]}{SEP}{sample.steps[0][2]}")
		
		return X_str, Y_str

	def _build_select_target(self, samples):
		Y_str = []

		for sample in samples:

			if sample.steps[0][2] == HAL:
				Y_str.append(str(sample.steps[-1][1]))
			else:
				Y_str.append(f"{sample.steps[0][1]}")
		
		return Y_str

	def _build_select_step_input(self, samples, step):
		X_str = []
		step_idx = int(step[1])

		for sample in samples:
			X_str.append(sample.solution_chain[step_idx])
		
		return X_str

	def _build_select_step_target(self, samples, step):
		Y_str = []
		step_idx = int(step[1])

		for sample in samples:
			Y_str.append(sample.sub_expr[step_idx])
		
		return Y_str

	def _build_solve_target(self, samples):
		return [str(sample.steps[-1][2]) for sample in samples]
		
	def _build_solve_atomic_target(self, samples):
		Y_str = []

		for sample in samples:
			if sample.steps[0][2] == HAL:	
				Y_str.append(f"{HAL}")
			else:
				Y_str.append(str(sample.steps[-1][2]))
		
		return Y_str

	def _build_combiner_target(self, samples):
		Y_str = []

		for sample in samples:
			sample_str = sample.to_string()
			
			if '+-' in sample_str:
				Y_str.append(sample_str.replace('+-', '-', 1))

			elif '-+' in sample_str:
				Y_str.append(sample_str.replace('-+', '-', 1))

			elif '--' in sample_str:
				Y_str.append(sample_str.replace('--', '+', 1))

			else:
				Y_str.append(sample.to_string(sample.steps[0][0]))

		return Y_str

	def _build_simplify_target(self, samples):
		combiner_target = self._build_combiner_target(samples)
		return [f"{sample}" for sample in combiner_target]
