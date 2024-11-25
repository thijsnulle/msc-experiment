import huggingface_hub, torch

from classes import GenerationOutput, GenerationMetrics
from llama_cpp import Llama

class LLM:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=131072,
            n_threads=24,
            logits_all=True,
            verbose=False,
        )

    def generate(self, prompt, max_tokens=256, stop_token='<|endoftext|>') -> GenerationOutput:
        output = self.llm(prompt, max_tokens=max_tokens, stop=stop_token, logprobs=1, temperature=0.25)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return GenerationOutput(
            text=output["choices"][0]["text"],
            tokens=output["choices"][0]["logprobs"]["tokens"],
            logprobs=output["choices"][0]["logprobs"]["token_logprobs"],
            finish_reason=output["choices"][0]["finish_reason"],
        )

    def __del__(self):
        try:
            if self.llm is not None:
                self.llm.close()
                self.llm = None
        except:
            pass

