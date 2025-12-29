from .base import ChatEngine, Message
from .openai_engine import AzureOpenAI
from .vllm import VLLM
from .huggingface_engine import HuggingFace

from pathlib import Path
from yaml import safe_load
from typing import Optional


def _all_subclasses(cls):
    return cls.__subclasses__() + [sub for child in cls.__subclasses__() for sub in _all_subclasses(child)]


def load_from_disk(path: Path, engine: Optional[str] = None) -> ChatEngine:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    if path.is_file():
        with path.open() as f:
            config = safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {path}")

    if engine is None: 
        engine = config.pop("engine", "HuggingFace")

    available_engines = {c.__name__: c for c in _all_subclasses(ChatEngine)}
    if engine not in available_engines:
        raise NotImplementedError(f"Model engine {engine} not supported. Supported engines are {list(available_engines.keys())}")
    return available_engines[engine].load_from_disk(path=path)
