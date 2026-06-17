from typing import Any
from vllm import LLM, SamplingParams
import os
import torch
from src import config

MODELS_PATH = {
    "qwen3-4b-instruct": f"{config.MODEL_PATH}/qwen3-4b-instruct"
}

class VLLMAdapter:

    def __init__(self, model_name: str, tensor_parallel_size: int, gpu_memory_utilization: float):

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0:
            raise RuntimeError("No visible CUDA device. Check CUDA_VISIBLE_DEVICES / job allocation / container GPU passthrough.")
        if tensor_parallel_size > n:
            raise ValueError(f"tensor_parallel_size={tensor_parallel_size} > visible_gpus={n}. Set tp=1 or expose more GPUs.")

        self.model_name = model_name
        self.llm = LLM(
            model=MODELS_PATH.get(model_name),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=32768,
        )

    def generate_batch(self, messages_list, temperature: float, **kwargs: Any):
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = 8192
        params = SamplingParams(**kwargs)
        outs = self.llm.chat(
            messages=messages_list,
            sampling_params=params,
        )
        output = []
        for out in outs:
            text = (out.outputs[0].text or "").strip()
            output.append(text)
        return output

    def generate_one(self, message, temperature: float, **kwargs: Any):
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = 8192
        params = SamplingParams(**kwargs)
        outputs = self.llm.chat(
            messages=[message], 
            sampling_params=params
        )
        return (outputs[0].outputs[0].text or "").strip()