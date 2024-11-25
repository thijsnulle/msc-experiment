from experiment import Experiment
from llm import LLM
from llama_cpp import Llama

if __name__ == '__main__':
    experiment = Experiment(
        small_model_path='models/Yi-Coder-1.5B',
        large_model_path='models/Yi-Coder-9B',
        dataset_path='data/dataset.jsonl',
    )

    experiment.start()

