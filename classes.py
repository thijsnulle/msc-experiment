import dataclasses, json, os, pathlib
import numpy as np

from typing import List, Optional

BASE_FOLDER = pathlib.Path('results')

class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

@dataclasses.dataclass
class CodeVerificationResult:
    code: str
    compilation_passed: bool
    time: float
    tests_passed: bool = False
    error: Optional[str] = None

@dataclasses.dataclass
class MCSResult:
    selected_lines: List[int]
    result: CodeVerificationResult

@dataclasses.dataclass
class ProblemFile:
    path: pathlib.Path
    no_lines: int

class ProblemFiles:
    def __init__(self, problem_id):
        self.folder = BASE_FOLDER / problem_id

        os.makedirs(self.folder, exist_ok=True)

        self.func_small = self._file(self.folder / 'func_small.jsonl')
        self.func_large = self._file(self.folder / 'func_large.jsonl')
        self.line_small = self._file(self.folder / 'line_small.jsonl')
        self.line_large = self._file(self.folder / 'line_large.jsonl')

    def current_iteration(self):
        return min([self.func_small.no_lines, self.func_large.no_lines, self.line_small.no_lines, self.line_large.no_lines])

    def get_file(self, is_func_level, is_small_llm) -> ProblemFile:
        if is_func_level:
            return self.func_small if is_small_llm else self.func_large

        return self.line_small if is_small_llm else self.line_large

    def _file(self, path: pathlib.Path) -> ProblemFile:
        with open(path, 'a+') as f:
            pass

        with open(path, 'r') as f:
            return ProblemFile(path=path, no_lines=len(f.readlines()))

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

