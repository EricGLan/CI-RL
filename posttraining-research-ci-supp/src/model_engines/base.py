import json
import time
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Literal, List, final, Optional, Type
from pathlib import Path
from yaml import safe_load
from functools import partial


@dataclass
class Message:
    """
    A class representing a message in a conversation.

    Attributes:
        role (Literal["system", "user", "assistant"]): The role of the message sender.
        content (str): The content of the message.
    """
    role: Literal["system", "user", "assistant"]
    content: str


class ChatEngine(ABC):
    @classmethod
    def load_from_disk(cls, path: Path):
        with path.open("r") as f:
            config = safe_load(f)
        config.pop("engine", None)
        return cls(**config)

    @abstractmethod
    def _gen(self, *, messages: Optional[List[Message]] = None, prompt: Optional[str] = None,
             response_format: Optional[Type] = None, **kwargs) -> Message:
        pass

    @final
    def gen(self, *, messages: Optional[List[Message]] = None, prompt: Optional[str] = None,
            response_format: Optional[Type] = None, **kwargs) -> Message:
        """
        Generate a response message based on the provided messages.

        Args:
            messages (List[Message]): A list of Message objects to base the response on.
            response_format (Optional[Type], optional): The expected format of the response. 
                Must be a dataclass if provided. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the response generation method.

        Raises:
            TypeError: If response_format is provided and is not a dataclass.

        Returns:
            Message: The generated response message.
        """
        if (messages is None) == (prompt is None):
            raise ValueError("Either messages or prompt must be provided, but not both.")

        if response_format is not None and not is_dataclass(response_format):
            raise TypeError("response_format must be a dataclass")
        return self._gen(messages=messages, prompt=prompt, **kwargs, response_format=response_format)
    
    def _batch_gen(
        self, messages_batch: List[List[Message]], prompt_batch: Optional[List[str]] = None,
        response_format: Optional[Type] = None, **kwargs
    ) -> List[Message]:
        if messages_batch is not None:
            return [self.gen(messages=messages, response_format=response_format, **kwargs) for messages in messages_batch]
        else:
            return [self.gen(prompt=prompt, response_format=response_format, **kwargs) for prompt in prompt_batch]
    
    @final
    def batch_gen(
        self, *, messages_batch: Optional[List[List[Message]]] = None, prompt_batch: Optional[List[str]] = None,
        response_format: Optional[Type] = None, **kwargs
    ) -> List[Message]:
        """
        Generates a batch of messages.

        Args:
            messages_batch (List[List[Message]]): A batch of messages, where each sublist represents a group of messages.
            response_format (Optional[Type], optional): The format of the response. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the batch generation method.

        Returns:
            List[Message]: A list of generated messages.
        """
        if (messages_batch is None) == (prompt_batch is None):
            raise ValueError("Either messages_batch or prompt_batch must be provided, but not both.")
   
        return self._batch_gen(messages_batch=messages_batch, prompt_batch=prompt_batch, **kwargs,
                               response_format=response_format)

    @abstractmethod
    def __getstate__(self, path: Path) -> dict:
        pass

    def save_to_disk(self, path: Path):
        path = Path(path)
        with path.open("w") as f:
            json.dump(self.__getstate__(), f)
