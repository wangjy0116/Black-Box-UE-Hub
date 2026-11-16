import os

from dotenv import load_dotenv

from .api_adapter import MultiFormatAdapter
from .vllm_adapter import VLLMAdapter


load_dotenv()

OPENROUTER_MODELS = {
    "qwen3-30b-instruct": "qwen/qwen3-30b-a3b-instruct-2507",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gpt-5-mini": "openai/gpt-5-mini",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
}
VLLM_MODELS = {"qwen3-4b-instruct": "qwen3-4b-instruct"}


def load_model(
    model_name: str,
    max_workers: int = 10,
    max_tokens: int = 8192,
):
    max_tokens = int(max_tokens)

    if model_name in OPENROUTER_MODELS:
        api_base = os.getenv("OPENROUTER_API_BASE")
        return MultiFormatAdapter(
            model_name=OPENROUTER_MODELS[model_name],
            api_base=api_base,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_workers=max_workers,
            max_tokens=max_tokens,
        )

    if model_name in VLLM_MODELS:
        api_base = os.getenv("VLLM_API_BASE")
        served_model_name = (
            os.getenv("VLLM_SERVED_MODEL_NAME") or VLLM_MODELS[model_name]
        )
        print(
            f"[MODEL] backend=vllm | base={api_base.rstrip('/')} "
            f"| served_model={served_model_name} | max_workers={max_workers}"
        )
        return VLLMAdapter(
            model_name=served_model_name,
            api_base=api_base,
            api_key=os.getenv("VLLM_API_KEY") or "EMPTY",
            max_workers=max_workers,
            max_tokens=max_tokens,
        )

    supported = ", ".join(sorted(OPENROUTER_MODELS | VLLM_MODELS))
    raise ValueError(
        f"Unsupported model {model_name!r}. Supported models: {supported}."
    )
