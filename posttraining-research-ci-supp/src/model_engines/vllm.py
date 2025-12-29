from typing import List, Optional, Type
from pathlib import Path
from copy import deepcopy

from .base import ChatEngine, Message


class VLLM(ChatEngine):
    def __init__(self, model: "vllm.LLM", tokenizer: Optional["transformers.PreTrainedTokenizer"] = None):
        from vllm import LLM, SamplingParams
        self.tokenizer = tokenizer
        self.model: LLM = model
        self.default_sampling_params: SamplingParams = SamplingParams()

    @classmethod
    def load_from_disk(cls, path: Path):
        from vllm import LLM
        from transformers import AutoTokenizer

        # if path is a config.json file then use the parent directory
        if path.name == "config.json":
            path = path.parent

        model = LLM(model=str(path))
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model=model, tokenizer=tokenizer)

    def _gen(
        self, *, messages: Optional[List[Message]], prompt: Optional[str] = None,
        response_format: Optional[Type] = None, max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Message:
        messages_batch, prompt_batch = None, None
        if messages is not None:
            messages_batch = [messages]
        if prompt is not None:
            prompt_batch = [prompt]

        responses = self._batch_gen(
            prompt_batch=prompt_batch, messages_batch=messages_batch, response_format=response_format, max_tokens=max_tokens,
            temperature=temperature, use_tqdm=False
        )
        return responses[0]
    
    def _batch_gen(
        self,
        *,
        messages_batch: Optional[List[List[Message]]] = None,
        prompt_batch: Optional[List[str]] = None,
        response_format: Optional[Type] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_tqdm: bool = True
    ) -> List[Message]:
        if response_format is not None:
            raise NotImplementedError("Response format not supported for VLLM engine")
        
        if (messages_batch is None) == (prompt_batch is None):
            raise ValueError("Either messages or prompts must be provided, but not both.")
        
        if messages_batch is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for processing messages.")
            
            prompt_batch = self.tokenizer.apply_chat_template(
                [[{"role": m.role, "content": m.content} for m in messages] for messages in messages_batch],
                add_generation_prompt=True,
                tokenize=False
            )

        sampling_params = deepcopy(self.default_sampling_params)
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        if temperature is not None:
            sampling_params.temperature = temperature

        outputs = self.model.generate(prompt_batch, sampling_params, use_tqdm=use_tqdm,)

        return [Message(role="assistant", content=output.outputs[0].text) for output in outputs]
    
    def __getstate__(self):
        raise NotImplementedError("VLLMEngine does not support __getstate__")
