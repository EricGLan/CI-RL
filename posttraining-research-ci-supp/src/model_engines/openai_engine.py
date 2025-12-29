import openai
import json
import os
import time
import logging
import mlflow

from pydantic import create_model, BaseModel, Field
from typing import List, Optional, Type, Any
from dataclasses import is_dataclass, fields, MISSING
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, ChainedTokenCredential, AzureCliCredential
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm_loggable.auto import tqdm

from .base import ChatEngine, Message
from .utils import GPNumWorkersSelector


logger = logging.getLogger(__name__)


def dataclass_to_pydantic(dc_cls: Type[Any]) -> Type[BaseModel]:
    if not is_dataclass(dc_cls):
        raise ValueError("Provided class is not a dataclass")

    pydantic_fields = {}
    for f in fields(dc_cls):
        field_args = dict()
        if "description" in f.metadata:
            field_args["description"] = f.metadata["description"]
        if f.default is not MISSING:
            field_args["default"] = f.default
        if f.default_factory is not MISSING:
            field_args["default_factory"] = f.default_factory
        pydantic_fields[f.name] = (f.type, Field(**field_args))

    model = create_model(dc_cls.__name__ + 'Model', **pydantic_fields)
    return model


class OpenAIEngineBase(ChatEngine):
    def _initialize(self, client: openai.OpenAI, model: str, batch_model: Optional[str] = None, strict: bool = False):
        self.client = client
        self.model = model
        self.batch_model = batch_model
        self.max_realtime_batchsize = 2_000
        self.strict = strict
        self.num_retries = 10
        self.num_workers_selector = GPNumWorkersSelector(
            candidates=[2, 4, 8, 16, 32, 64],
            beta=2.0,
            window_size=30,
        )

    def _gen(
        self, messages: List[Message], prompt: Optional[str] = None, response_format: Optional[Type] = None,
        max_tokens: Optional[int] = None, temperature: Optional[None] = None
    ) -> Message:
        if prompt is not None:
            raise ValueError("prompt is not supported for OpenAIEngineBase")

        messages=[{"role": m.role, "content": m.content} for m in messages]

        num_tries = 0
        try_again = True
        while try_again:
            try:
                num_tries += 1

                extra_kwargs = {}
                if max_tokens is not None:
                    extra_kwargs["max_tokens"] = max_tokens
                if temperature is not None:
                    extra_kwargs["temperature"] = temperature
    
                if response_format is not None:
                    if not is_dataclass(response_format):
                        raise TypeError("response_format must be a dataclass")
                    
                    PydanticModel = dataclass_to_pydantic(response_format)

                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=PydanticModel,
                        **extra_kwargs,
                    )
                    response.choices[0].message.content = response.choices[0].message.parsed.model_dump_json()
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **extra_kwargs,
                    )
            except openai.BadRequestError as e:
                response = self._handle_bad_request_error(e=e, messages=messages, num_tries=num_tries)
            except openai.RateLimitError as e:
                response = self._handle_rate_limit_error(e=e, messages=messages, num_tries=num_tries)
            try_again = response is None

        return Message(role=response.choices[0].message.role, content=response.choices[0].message.content)

    def _handle_rate_limit_error(
        self, e: openai.RateLimitError, messages: List[Message], num_tries: int
    ) -> Optional[openai.types.chat.ChatCompletion]:
        logger.error(f"Rate limit error: {e}.")
        if num_tries >= self.num_retries:
            logger.error(f"Max retries reached: {self.num_retries}. Request: {messages}")
            raise e
        logger.info(f"Retrying in 10 seconds...")
        time.sleep(10)
        return None
    
    def _handle_bad_request_error(
        self, e: openai.BadRequestError, messages: List[Message], num_tries: int
    ) -> Optional[openai.types.chat.ChatCompletion]:
        logger.error(f"Bad request: {e}. Request: {messages}")
        if self.strict:
            raise e
        
        if num_tries >= self.num_retries:
            logger.error(f"Max retries reached: {self.num_retries}. Request: {messages}")
            raise e

        if e.code == "invalid_prompt":
            return openai.types.chat.ChatCompletion.construct(**{
                "model": self.model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": e.message},
                        "finish_reason": "content_filter",
                    },
                ]
            })
        raise e
    
    def _batch_gen_realtime(
        self, messages_batch: List[List[Message]], response_format: Optional[Type] = None, max_tokens: Optional[int] = None,
        temperature: Optional[None] = None
    ) -> List[Message]:
        max_realtime_batchsize = max(self.num_workers_selector.candidates)
        if len(messages_batch) > max_realtime_batchsize:
            # If the batch size is significantly larger than the maximum number of threads, reduce the batch size
            responses = []
            with tqdm(total=len(messages_batch)) as pbar:
                for i in range(0, len(messages_batch), max_realtime_batchsize):
                    pbar.update(len(messages_batch[i:i+max_realtime_batchsize]))
                    responses.extend(
                        self._batch_gen_realtime(
                            messages_batch[i:i+max_realtime_batchsize], response_format=response_format, max_tokens=max_tokens,
                            temperature=temperature
                        )
                    )
            return responses

        num_samples = len(messages_batch) 
        thread_count = self.num_workers_selector.get_num_workers()
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            responses = list(executor.map(
                partial(self._gen, response_format=response_format, max_tokens=max_tokens, temperature=temperature),
                messages_batch,
            ))
        throughput = num_samples / (time.time() - start_time)
        logger.info(f"{thread_count} threads produced throughput: {throughput:.2f} samples/s")
        mlflow.log_metric("openai_api_samples_per_second", throughput)
        mlflow.log_metric("openai_api_parallel_calls", thread_count)
        self.num_workers_selector.update(num_workers=thread_count, throughput=throughput)
        return responses
    
    def _batch_gen(
        self, messages_batch: List[List[Message]], prompt_batch: Optional[List[str]] = None,
        response_format: Optional[Type] = None, max_tokens: Optional[int] = None, temperature: Optional[None] = None
    ) -> List[Message]:
        if prompt_batch is not None:
            raise ValueError("prompt_batch is not supported for OpenAIEngineBase")

        if self.batch_model is None:
            return self._batch_gen_realtime(
                messages_batch, response_format=response_format, max_tokens=max_tokens, temperature=temperature
            )

        if len(messages_batch) < self.max_realtime_batchsize:
            return self._batch_gen_realtime(
                messages_batch, response_format=response_format, max_tokens=max_tokens, temperature=temperature
            )
        
        additional_body_kwargs = dict()
        if response_format is not None:
            if not is_dataclass(response_format):
                raise TypeError("response_format must be a dataclass")
            PydanticModel = dataclass_to_pydantic(response_format)
            # Not entirely sure where else we can import this from.
            additional_body_kwargs["response_format"] = openai.lib._parsing._completions.type_to_response_format_param(PydanticModel)
        if max_tokens is not None:
            additional_body_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            additional_body_kwargs["temperature"] = temperature

        # Prepare batch data as list of records.
        batch_data = []
        for idx, messages in enumerate(messages_batch):
            request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self.batch_model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    **additional_body_kwargs,
                },
            }
            batch_data.append(request)
        
        # Write data to a temporary JSONL file.
        with TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / "batch_data.jsonl"
            with tmp_file.open("w") as f:
                for record in batch_data:
                    f.write(json.dumps(record) + "\n")

            # Upload the file with purpose "batch".
            with open(tmp_file, "rb") as f:
                file_resp = self.client.files.create(file=f, purpose="batch")
            file_id = file_resp.id

        # Create a batch request.
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="chat/completions",
            completion_window="24h"
        )
        batch_id = batch_response.id

        # Poll until the batch is complete.
        status = batch_response.status
        while status not in ("completed", "failed", "cancelled"):
            time.sleep(60)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            logger.info(f"Batch id {batch_id}: status: {status}")

        if status == "failed":
            for error in batch_response.errors:
                logger.error(f"Error code {error.code}: {error.message}")
            raise RuntimeError("Batch failed")
        
        if status == "cancelled":
            raise RuntimeError("Batch cancelled")
        
        responses_batch = self.retrieve_responses_batch(batch_id)
        return [Message(role=resp.choices[0].message.role, content=resp.choices[0].message.content) for resp in responses_batch]
   
    def retrieve_responses_batch(self, batch_id: str) -> List[openai.types.chat.ChatCompletion]:
        # Poll until the batch is complete.
        batch_response = self.client.batches.retrieve(batch_id)
        status = batch_response.status
        while status not in ("completed", "failed", "cancelled"):
            time.sleep(60)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            logger.info(f"Batch id {batch_id}: status: {status}")

        if status == "failed":
            for error in batch_response.errors:
                logger.error(f"Error code {error.code}: {error.message}")
            raise RuntimeError("Batch failed")
        
        if status == "cancelled":
            raise RuntimeError("Batch cancelled")
        
        error_file_id = batch_response.error_file_id
        file_content = self.client.files.content(error_file_id).text.strip()
        if len(file_content) > 0:
            raw_lines = file_content.split("\n")
            print("Error file content:")
            for line in raw_lines:
                print(line)
            raise RuntimeError("Batch failed with errors")

        output_file_id = batch_response.output_file_id or batch_response.error_file_id
        file_content = self.client.files.content(output_file_id)
        raw_lines = file_content.text.strip().split("\n")
        
        # Parse responses and sort them by custom_id.
        responses = []
        for line in raw_lines:
            resp_dict = json.loads(line)
            responses.append((int(resp_dict.get("custom_id")), resp_dict))
        responses.sort(key=lambda x: x[0])

        # Convert each response to a Message.
        final_responses = []
        for _, resp in responses:
            final_responses.append(
                openai.types.chat.ChatCompletion.construct(**(resp["response"]["body"]))
            )
        return final_responses


class AzureOpenAI(OpenAIEngineBase):
    def __init__(
        self, deployment: str, endpoint: str, api_version: str, use_azure_ad_token_provider: bool = True,
        batch_deployment: Optional[str] = None
    ):
        self._initialize(
            deployment=deployment,
            endpoint=endpoint,
            api_version=api_version,
            use_azure_ad_token_provider=use_azure_ad_token_provider,
            batch_deployment=batch_deployment
        )

    def _initialize(self, deployment: str, endpoint: str, api_version: str, use_azure_ad_token_provider: bool,
                    batch_deployment: Optional[str]):
        azure_ad_token_provider = None
        if use_azure_ad_token_provider:
            os.environ["AZURE_CLIENT_ID"] = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID', os.environ.get('AZURE_CLIENT_ID', ""))

            credentials = ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(),
            )

            azure_ad_token_provider = get_bearer_token_provider(
                credentials, "https://cognitiveservices.azure.com/.default"
            )
        self.deployment = deployment
        self.batch_deployment = batch_deployment
        self.endpoint = endpoint
        self.api_version = api_version
        self.use_azure_ad_token_provider = use_azure_ad_token_provider

        super()._initialize(
            client=openai.AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                azure_ad_token_provider=azure_ad_token_provider,
                timeout=60,
            ),
            model=self.deployment,
            batch_model=self.batch_deployment,
            strict=True
        )

    def __getstate__(self):
        return {
            "type": "AzureOpenAI",
            "deployment": self.deployment,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "use_azure_ad_token_provider": self.use_azure_ad_token_provider,
            "batch_deployment": self.batch_deployment,
        }

    def __setstate__(self, state):
        self._initialize(
            deployment=state["deployment"],
            endpoint=state["endpoint"],
            api_version=state["api_version"],
            use_azure_ad_token_provider=state["use_azure_ad_token_provider"],
            batch_deployment=state["batch_deployment"],
        )
