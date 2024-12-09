import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast, contextlib, dataclasses, gc, glob, io, json, multiprocessing, re, signal, time, unittest

from classes import DataclassJSONEncoder, InputPrompt, GenerationMetrics, GenerationOutput, GenerationResult
from collections import defaultdict
from data_processor import DataProcessor
from pathlib import Path
from typing import Optional

def test_handler(signum, frame):
    raise TimeoutError()

@dataclasses.dataclass
class CodeVerificationResult:
    code: str
    compilation_passed: bool
    time: float
    tests_passed: bool = False
    error: Optional[str] = None

RESULTS_DIR = '/Users/thijs/Documents/Git/msc-experiment/results'

SKIPPED_PROBLEMS_FILE = Path('/Users/thijs/Documents/Git/msc-experiment/tests/skipped_problems.txt')
VERIFICATIONS_DIR_PATH = Path('/Users/thijs/Documents/Git/msc-experiment/tests/results')

def load_data(file_name):
    output = defaultdict(list)

    for file in glob.glob(f'{RESULTS_DIR}/**/{file_name}.jsonl'):
        problem_id = re.search(r'(\d+)', file)[0]

        with open(file) as f:
            for line in f.read().splitlines():
                result = GenerationResult(**json.loads(line))
                result.outputs = [GenerationOutput(**o) for o in result.outputs]
                result.metrics = GenerationMetrics(**result.metrics)

                output[problem_id].append(result)

    return output


class CodeVerifier:
    def verify(self, problem, generation_results):
        if problem.id != '95':
            return

        gc.collect()

        self._load_cache(problem)

        with open(SKIPPED_PROBLEMS_FILE, 'a+'):
            pass

        with open(SKIPPED_PROBLEMS_FILE, 'r') as f:
            if problem.id in f.readlines():
                return

        test_lines = problem.reference.test_code.splitlines()
        test_lines.insert(1, 'from unittest.mock import patch\npatch("matplotlib.pyplot.show").start()')
        problem.reference.test_code = '\n'.join(test_lines)

        if not bool(self.cache):
            # if tests of problem take >2 seconds, skip for now.
            signal.signal(signal.SIGALRM, test_handler)
            signal.alarm(1)

            try:
                with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                    verification_result = self._verify(problem, InputPrompt('', 0, 100), '')
            except:
                verification_result = None

            signal.alarm(0)

            if verification_result and verification_result.error:
                print(verification_result.error)

            if not verification_result or verification_result.error == 'TimeoutError' or verification_result.time > 2.0:
                print(f'Problem {problem.id} --- Skipped')

                with open(SKIPPED_PROBLEMS_FILE, 'a+') as f:
                    f.write(problem.id)
                    f.write('\n')

                return

        code_lines = [x for x in problem.reference.code.splitlines() if x.strip()]

        problem_path = VERIFICATIONS_DIR_PATH / problem.id
        problem_path.mkdir(parents=True, exist_ok=True)

        cache_hits = 0
        non_cache_hits = 0

        for result in generation_results:
            for i, output in enumerate(result.outputs):
                # TODO: perform actual data processing to remove this.

                line_prompt = problem.line_prompts[i]

                code = output.text if '<|endoftext|>' not in output.text else output.text.split('<|endoftext|>')[0]

                if code in self.cache[line_prompt.line_index]:
                    cache_hits += 1
                    continue
                else:
                    non_cache_hits += 1

                with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                    verification_result = self._verify(problem, line_prompt, code)

                self.cache[line_prompt.line_index][code] = verification_result

                with open(problem_path / f'{line_prompt.line_index}.jsonl', 'a+') as f:
                    f.write(json.dumps(verification_result, cls=DataclassJSONEncoder))
                    f.write('\n')

                gc.collect()

        tests_passed = 0
        tests_total = 0

        for _, line_index_values in self.cache.items():
            for result in line_index_values.values():
                tests_total += 1
                tests_passed += 1 if result.tests_passed else 0

        print(f'Problem {problem.id}\t- Cache Hit %: {cache_hits/(cache_hits+non_cache_hits):.2f},\tTest Pass %: {tests_passed/tests_total:.2f}') 


    def _verify(self, problem, line_prompt, code):
        t0 = time.time()

        line_index = line_prompt.line_index
        char_index = line_prompt.char_index

        compiled_code, did_compile, error = self._compile(problem, line_index, char_index, code)

        if not did_compile:
            return CodeVerificationResult(code=code, compilation_passed=False, error=error, time=time.time() - t0)

        tests_passed, error = self._test(compiled_code)

        if not tests_passed:
            return CodeVerificationResult(code=code, compilation_passed=True, tests_passed=False, error=error, time=time.time() - t0)

        return CodeVerificationResult(code=code, compilation_passed=True, tests_passed=True, time=time.time() - t0)


    def _compile(self, problem, line_index, char_index, code):
        """
        returns: code, compilation_passed, error
        """
        solution_lines = problem.reference.complete_code.splitlines()
        solution_lines[line_index] = solution_lines[line_index][:char_index] + code
        solution_lines.extend(problem.reference.test_code.splitlines())

        solution = '\n'.join(solution_lines)

        try:
            abstract_syntax_tree = ast.parse(solution)
            code = compile(abstract_syntax_tree, filename='<string>', mode='exec')
        except Exception as e:
            return None, False, type(e).__name__

        return code, True, None

    def _test(self, code):
        """
        returns: tests_passed, exception, time
        """

        buffer = io.StringIO()
        context = {}

        signal.signal(signal.SIGALRM, test_handler)
        signal.alarm(1)

        try:
            exec(code, context)

            runner = unittest.TextTestRunner(stream=buffer, failfast=True, verbosity=0)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(context['TestCases'])

            test_result = runner.run(suite)
        except Exception as e:
            return False, str(e)

        signal.alarm(0)

        if test_result.errors:
            exception = test_result.errors[0][1].splitlines()[-1].split(':')[0]

            return False, exception

        return True, None

    def _load_cache(self, problem):
        self.cache = defaultdict(lambda: defaultdict(set))

        problem_path = VERIFICATIONS_DIR_PATH / problem.id

        if not problem_path.exists():
            return

        for problem_file in glob.glob(f'{problem_path}/*.jsonl'):
            line_index = int(re.search('(\d+)\.jsonl', problem_file)[1])

            with open(problem_file, 'r') as f:
                for verification_result_str in f.readlines():
                    result = CodeVerificationResult(**json.loads(verification_result_str))

                    self.cache[line_index][result.code] = result
        

if __name__ == '__main__':
    code_verifier = CodeVerifier()

    line_small = load_data('line_small')

    cnt = 0

    for generation_results in line_small.values():
        counts = defaultdict(set)

        for generation_result in generation_results:
            for i, output in enumerate(generation_result.outputs):
                counts[i].add(output.text)
    
        for c in counts.values():
            cnt += len(c)

    print(cnt)

    """

    problems = DataProcessor.load(input_file_path='../data/test-dataset.jsonl')

    for problem_id, generation_results in line_small.items():
        problem = problems[int(problem_id)]       

        result = code_verifier.verify(problem, generation_results)

    #result = code_verifier.verify(problem, generation_results=line_small)

    #print(result)

    """

    """
    try:
        abstract_syntax_tree = ast.parse("def main():\n    d\n    return 5\n")
    except Exception as e:
        exception_type = type(e).__name__

    print(problem.reference.complete_code)

    code = compile(abstract_syntax_tree, filename='<string>', mode='exec')

    result = exec(code)

    print(result)
    """

"""
@dataclasses.dataclass
class InputPrompt:
    prompt: str
    line_index: int
    char_index: int

@dataclasses.dataclass
class ReferenceSolution:
    code: str
    complete_code: str
    test_code: str

@dataclasses.dataclass
class Problem:
    id: str
    reference: ReferenceSolution
    func_prompt: InputPrompt
    line_prompts: list[InputPrompt]

    def is_done(self, N):
        self.files = ProblemFiles(self.id)

        return (
            self.files.func_small.no_lines == N and
            self.files.func_large.no_lines == N and
            self.files.line_small.no_lines == N and
            self.files.line_large.no_lines == N
        )

@dataclasses.dataclass
class GenerationMetrics:
    energy: float
    energy_per_token: float
    time: float
    time_per_token: float

@dataclasses.dataclass
class GenerationOutput:
    text: str
    tokens: list[str]
    logprobs: list[float]
    finish_reason: Optional[str] # "stop" | "length"

@dataclasses.dataclass
class GenerationResult:
    outputs: list[GenerationOutput]
    metrics: GenerationMetrics
"""

