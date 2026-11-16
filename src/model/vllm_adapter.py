from .api_adapter import MultiFormatAdapter


class VLLMAdapter(MultiFormatAdapter):
    """Client for a separately launched OpenAI-compatible vLLM server."""

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "EMPTY",
        max_workers: int = 10,
        timeout: int = 600,
        max_tokens: int = 8192,
        max_retries: int = 2,
    ):
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            max_workers=max_workers,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_sleep=1.0,
            max_sleep=5.0,
        )
