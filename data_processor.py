import json, re, os

from classes import Problem, ReferenceSolution, InputPrompt, DataclassJSONEncoder

class DataProcessor:
    trigger_statements = [
        'assert', 'raise', 'del', 'lambda', 'yield', 'return', 'while',
        'for', 'if', 'else', 'elif', 'global', 'in', 'and', 'not', 'or',
        'is', 'with', 'except', '.', '+=', '+', '-=', '-', '*', '/', 
        '%', '**', '<<', '>>', '&', '|', '^', '==', '!=', '<=', '>=', '=',
        '<', '>', ';', ',', '(', '{', '~', ')', '}', ':', '@',
    ]

    @staticmethod
    def process(input_file_path, output_file_path):
        with open(input_file_path) as input_file:
            file_content = input_file.read()

        data = [json.loads(content) for content in file_content.splitlines()]
        problems = [DataProcessor.process_data_point(id, data_point) for id, data_point in enumerate(data)]

        if os.path.isfile(output_file_path):
            os.remove(output_file_path)
            #raise OSError(f'File `{output_file_path}` already exists.')

        with open(output_file_path, 'a+') as output_file:
            for problem in problems:
                output_file.write(json.dumps(problem, cls=DataclassJSONEncoder))
                output_file.write('\n')

    @staticmethod
    def load(input_file_path) -> list[Problem]:
        with open(input_file_path) as input_file:
            file_content = input_file.read()

        problems = []

        for line in file_content.splitlines():
            problem = Problem(**json.loads(line))
            problem.id = str(problem.id)
            problem.reference = ReferenceSolution(**problem.reference)
            problem.func_prompt = InputPrompt(**problem.func_prompt)
            problem.line_prompts = [InputPrompt(**i) for i in problem.line_prompts]
            problems.append(problem)

        return problems

    def process_data_point(id, data_point) -> Problem:
        prompt = data_point['complete_prompt']
        solution = data_point['canonical_solution']
        test_code = data_point['test']

        total_lines_prompt = len(prompt.splitlines())

        lines = [l for l in solution.splitlines() if l.strip()]
        line_and_char_indexes = map(lambda l: DataProcessor.map_to_indexes(*l), enumerate(lines))
        line_and_char_indexes = filter(lambda l: l, line_and_char_indexes)

        reference = ReferenceSolution(code=solution, complete_code=prompt + solution, test_code=test_code)
        func_prompt = InputPrompt(prompt=f'{prompt}    ', line_index=total_lines_prompt, char_index=4)

        line_prompts = [
            InputPrompt(
                prompt=prompt + '\n'.join(lines[:line_index]) + '\n' + lines[line_index][:char_index],
                line_index=line_index + total_lines_prompt,
                char_index=char_index,
            ) for line_index, char_index in line_and_char_indexes
        ]

        return Problem(
            id=str(id),
            reference=reference,
            func_prompt=func_prompt,
            line_prompts=line_prompts,
        )

    def map_to_indexes(line_index, line):
        stripped_line = line.strip()

        # TODO: what about this SQL-statement?
        if stripped_line.startswith('CREATE TABLE'):
            return None

        if stripped_line.startswith(('#', '"', 'import ', 'from ', 'def ', 'else:', 'try:', 'finally:', 'except:')):
            return None 

        statements = re.split(r'([ \(\):=@\.,])', stripped_line)

        if len(statements) == 1:
            return None

        statement_indexes = [statements.index(s) for s in DataProcessor.trigger_statements if s in statements]

        if len(statement_indexes) == 0:
            return None

        first_statement = statements[min(statement_indexes)]
        char_index = line.index(first_statement) + len(first_statement)

        if (first_statement == '+' and '+=' in line) or (first_statement == '-' and '-=' in line):
            char_index += 1
        
        if first_statement == 'for':
            char_index = len(re.match(r'(\s+for \w+(, \w+)*)', line).group(1))

        if char_index == len(line): # E.g. only closing parenthesis, closing bracket etc.
            return None

        return line_index, char_index

if __name__ == '__main__':
    DataProcessor.process(input_file_path='data/dataset.json', output_file_path='data/processed-dataset.jsonl')
    problems = DataProcessor.load(input_file_path='data/processed-dataset.jsonl')

    for problem in problems:
        break

# Problems w/ failing tests
# [49, 89, 101, 111, 115, 128, 157, 177, 198, 200, 221, 237, 242, 245, 276, 296, 334, 383, 416, 501, 568, 593, 634, 686, 699, 734, 736, 774, 940, 1005, 1028, 1109]:

# Problems that break subsequent problems
# [790]

# Problems that have plt.show()
# [119, 134]

# Problems w/ infinite loop
# [205, 363]

# Problems with unknown module
# [372, 583, 964]

