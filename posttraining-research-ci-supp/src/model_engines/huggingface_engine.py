import torch
import re

from typing import List, Optional, Type
from pathlib import Path

from .base import ChatEngine, Message


class HuggingFace(ChatEngine):
    def __init__(self, model: "transformers.PreTrainedModel", tokenizer: "transformers.PreTrainedTokenizer"):
        from transformers import PreTrainedModel, PreTrainedTokenizer
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer

    @classmethod
    def load_from_disk(cls, path: Path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # if path is a config.json file then use the parent directory
        if path.name == "config.json":
            path = path.parent

        device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model=model, tokenizer=tokenizer)

    def _gen(
        self, *, messages: Optional[List[Message]], prompt: Optional[str] = None,
        response_format: Optional[Type] = None
    ) -> Message:
        assert prompt is None, "prompt is not supported for HuggingFaceEngine"

        if response_format is not None:
            raise NotImplementedError("response_format is not supported for HuggingFaceEngine")
        
        sss = """# Instructions

Within <think> and </think> tags, provide your reasoning based on contextual integrity. \
Finally, within <answer> and </answer> tags, provide your final answer."""
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content.removesuffix("Answer: ")+sss} for m in messages],
            add_generation_prompt=True,
            tokenize=False
        ) + "<think>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)

        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        response = response.removeprefix(prompt)
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            extracted_text = match.group(1)
            response = extracted_text.strip("\n").strip()
        else:
            print("No match found in the response.")
            print("Response: ", response)
            response = ""

        return Message(role="assistant", content=response)
    
    def __getstate__(self):
        raise NotImplementedError("HuggingFaceEngine does not support __getstate__")
