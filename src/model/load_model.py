import os
from .vllm_adapter import VLLMAdapter
from .api_adapter import MultiFormatAdapter
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE")

openrouter_models = {"qwen3-30b-instruct": "qwen/qwen3-30b-a3b-instruct-2507",
                     "deepseek-v3.2": "deepseek/deepseek-v3.2",
                     "gpt-5-mini": "openai/gpt-5-mini"}

def load_model(model_name: str, max_workers: int = 10):
    if model_name in openrouter_models:  
        return  MultiFormatAdapter(
            model_name = openrouter_models[model_name],
            api_base = api_base,
            api_key = api_key,
            max_workers = max_workers
        )

    return VLLMAdapter(
        model_name = model_name,
        tensor_parallel_size = 1 ,
        gpu_memory_utilization = 0.8
    )