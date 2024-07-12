import re
import warnings


class GPT4OutputParser:

    def __init__(self):
        self.output_type = None
        self.error_counter = 0

    def _filter_matches(self, matches):
        if isinstance(matches[0], tuple):
            matches = [(m[0].strip(), m[1].strip(), m[2].strip()) for m in matches]
            matches = [match for match in matches if match != ('', '', '')]
        return matches

    def _simple_parse_outputs(self, outputs):
        outputs = self._preprocessing_step(outputs)
        try:
            matches = self.simple_output_re.findall(outputs)
            matches = self._filter_matches(matches)
            match = matches[-1]
        except IndexError:
            warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
            match = '-100'
            self.error_counter += 1

        if isinstance(match, tuple):
            match = match[0]
            if match == '':
                warnings.warn(f"Match is empty for GPT-4 output: {outputs}")
                match = '-100'
                self.error_counter += 1

        if self.output_type == int:
            return int(match)
        elif self.output_type == str:
            return match
        
    def parse_outputs(self, outputs):
        return self._simple_parse_outputs(outputs)

class GPT4ListopsOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r'\d')
        self.output_type = int

    def _preprocessing_step(self, output):
        output = output.replace('modulo 10', '')
        output = output.replace('mod 10', '')
        output = output.replace('Modulo 10', '')
        output = output.replace('Mod 10', '')
        output = output.replace('modulus 10', '')
        output = output.replace('Modulus 10', '')
        return output

class GPT4ArithmeticOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r'\-{0,1}\d\d*')
        self.output_type = int

    def _preprocessing_step(self, output):
        output = output.replace('modulo -100', '')
        output = output.replace('Modulo -100', '')
        output = output.replace('(mod -100)', '')
        output = output.replace('mod -100', '')
        output = output.replace('Mod -100', '')
        output = output.replace('modulo 100', '')
        output = output.replace('Modulo 100', '')
        output = output.replace('(mod 100)', '')
        output = output.replace('mod 100', '')
        output = output.replace('Mod 100', '')
        output = output.replace('modulus 100', '')
        output = output.replace('Modulus 100', '')
        return output

class GPT4AlgebraOutputParser(GPT4OutputParser):

    def __init__(self):
        super().__init__()
        self.simple_output_re = re.compile(r'([+-]*[0-9]*[0-9]*[* ]*[abxy*]*[ ]*([(]([-+0-9]|[-abxy])+)*[abxy*]*[ ]*[+-]*[ ]*[0-9]*[* ]*[abxy* ]*[/0-9]*[)]*[abxy*]*)')
        self.output_type = str

    def _preprocessing_step(self, output):
        output = output.replace('modulo -100', '')
        output = output.replace('Modulo -100', '')
        output = output.replace('(mod -100)', '')
        output = output.replace('mod -100', '')
        output = output.replace('Mod -100', '')
        output = output.replace('modulo 100', '')
        output = output.replace('Modulo 100', '')
        output = output.replace('(mod 100)', '')
        output = output.replace('mod 100', '')
        output = output.replace('Mod 100', '')
        output = output.replace('modulus 100', '')
        output = output.replace('Modulus 100', '')
        return output
        
def build_parser(task_name):
    if task_name == 'algebra':
        return GPT4AlgebraOutputParser()
    elif task_name == 'arithmetic':
        return GPT4ArithmeticOutputParser()
    elif task_name == 'listops':
        return GPT4ListopsOutputParser()
    else:
        assert False, f"Wrong task name {task_name}"
