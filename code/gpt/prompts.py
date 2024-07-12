import random


class PromptBuilder:

    def __init__(self):
        self.task_name = None
        self.blueprints = None
        
    def _build_prompt_no_examples(self, batch, prompt_type):
        return [self.blueprints[prompt_type].format(sample) for sample in batch]

    def build_prompt(self, batch, prompt_type):
        if prompt_type == 'self_consistency':
            prompt_type = 'zero_shot_cot'

        return self._build_prompt_no_examples(batch, prompt_type)
        

class ArithmeticPromptBuilder(PromptBuilder):

    def __init__(self):
        super().__init__()
        self.task_name = 'arithmetic'
        self.blueprints = {
            "zero_shot_cot": "Q: Solve the following arithmetic expression computing the modulo 100 of each intermediate value "
                             "if it's positive, and the modulo -100 if it's negative:\n{}.\n\n"
                             "A: Let's think step-by-step.",
        }


class AlgebraPromptBuilder(PromptBuilder):
        
    def __init__(self):
        super().__init__()
        self.task_name = 'algebra'
        self.blueprints = {
            "zero_shot_cot": "Q: Simplify the following algebraic expression, computing the modulo 100 of the numerical coefficient of each "
                             "intermediate value if it's positive, and the modulo -100 if it's negative:\n{}.\n"
                             "A: Let's think step-by-step.",
        }


class ListopsPromptBuilder(PromptBuilder):

    def __init__(self):
        super().__init__()
        self.task_name = 'listops'
        self.blueprints = {
            "zero_shot_cot": "Q: MIN, MAX and SM are operators on lists of single-digit integers which have the semantics of "
                             "minimum, maximum and sum modulo 10, respectively. "
                             "Solve the following expression involving these operators:\n{}.\n\n"
                             "A: Let's think step-by-step.",
        }


def get_prompt_builder(task_name):
    if task_name == 'listops':
        return ListopsPromptBuilder()
    elif task_name == 'arithmetic':
        return ArithmeticPromptBuilder()
    elif task_name == 'algebra':
        return AlgebraPromptBuilder()
    else:
        assert False, f"Wrong task name: {task_name}"
