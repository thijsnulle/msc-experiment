import dataclasses, gc, itertools, json, os, pathlib, random, signal

from classes import GenerationMetrics, GenerationResult, InputPrompt, Problem, ProblemFiles, DataclassJSONEncoder
from data_processor import DataProcessor
from llm import LLM
from pyEnergiBridge.api import EnergiBridgeRunner
from time import perf_counter

N = 30
SEED = 20891019142112125

def timeout_handler(signum, frame):
    raise TimeoutError(f'Timeout occurred, signum: {signum}, frame: {frame}')

class Experiment:
    def __init__(self, small_model_path, large_model_path, dataset_path):
        self.small_llm = LLM(small_model_path) 
        self.large_llm = LLM(large_model_path)
        self.problems = DataProcessor.load(input_file_path=dataset_path)
        self.num_problems = len(self.problems)
        self.runner = EnergiBridgeRunner(verbose=False)

        random.seed(SEED)
        random.shuffle(self.problems)

    def start(self):
        while True:
            selected_problem = self._select_problem()

            if selected_problem is None:
                break

            self.run(selected_problem)

    def run(self, problem):
        self.small_llm.llm.tokenize(str.encode(problem.reference.complete_code))
        self.large_llm.llm.tokenize(str.encode(problem.reference.complete_code))

        #self.small_llm.generate(problem.reference.complete_code, max_tokens=1)
        #self.large_llm.generate(problem.reference.complete_code, max_tokens=1)

        while not problem.is_done(N):
            available_prompts = self._get_available_prompts(problem)

            while len(available_prompts) > 0:
                raw_prompt, llm = available_prompts.pop()

                is_func_level = isinstance(raw_prompt, InputPrompt)
                is_small_llm = llm == self.small_llm

                prompts = [raw_prompt] if is_func_level else raw_prompt
                stop_token = ['<|endoftext|>', '\n\nprint', '\n\ndef', '\n\nif', '\n\n#'] if is_func_level else ['<|endoftext|>', '\n']

                self.runner.start()

                outputs = [llm.generate(p.prompt, stop_token=stop_token) for p in prompts]

                signal.signal(signal.SIGALRM, timeout_handler)

                signal.alarm(1)
                energy, time = self.runner.stop()
                signal.alarm(0)

                if not energy or not time:
                    print(f'Energy: {energy}, time: {time}')
                    continue

                total_tokens = sum(len(o.tokens) for o in outputs)

                metrics = GenerationMetrics(
                    energy=energy,
                    energy_per_token=energy / total_tokens,
                    time=time,
                    time_per_token=time / total_tokens,
                )

                generation_result = GenerationResult(outputs, metrics)

                file = problem.files.get_file(is_func_level, is_small_llm)

                with open(file.path, 'a+') as f:
                    f.write(json.dumps(generation_result, cls=DataclassJSONEncoder))
                    f.write('\n')

                #print(f'Function Level: {is_func_level}, Small LLM: {is_small_llm}, Energy: {energy}, Time: {time}')

                gc.collect()

            #print()

    def _select_problem(self) -> Problem:
        self.problems = list(itertools.dropwhile(lambda p: p.is_done(N), self.problems))

        print(f'Problem {self.num_problems - len(self.problems) + 1}/{self.num_problems}')

        if len(self.problems) > 0:
            return self.problems[0]

        return None

    def _get_available_prompts(self, problem):
        available_prompts = []
        iteration = problem.files.current_iteration()

        if problem.files.func_small.no_lines == iteration:
            available_prompts.append((problem.func_prompt, self.small_llm))

        if problem.files.func_large.no_lines == iteration:
            available_prompts.append((problem.func_prompt, self.large_llm))

        if problem.files.line_small.no_lines == iteration:
            available_prompts.append((problem.line_prompts, self.small_llm))

        if problem.files.line_large.no_lines == iteration:
            available_prompts.append((problem.line_prompts, self.large_llm))

        random.shuffle(available_prompts)

        return available_prompts

