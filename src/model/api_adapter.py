from typing import Any, Dict, List, Optional
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


class MultiFormatAdapter:
    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 10,
        timeout: int = 60,
        max_tokens: int = 8192,
        max_retries: int = 10,
        base_sleep: float = 1.0,
        max_sleep: float = 20.0,
    ):
        if not api_key:
            raise ValueError("api_key is required.")

        self.openrouter_model_name = model_name
        self.timeout = int(timeout)
        self.max_workers = int(max_workers)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.base_sleep = float(base_sleep)
        self.max_sleep = float(max_sleep)

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.timeout,
        )

    def _is_retryable(self, e: Exception):
        s = str(e).lower()
        return any(k in s for k in [
            "timed out", "timeout",
            "rate limit", "429",
            "503", "502", "500", "504",
            "connection", "reset", "overloaded", "service unavailable",
        ])

    def generate_one(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ):
        last_err = None

        for attempt in range(self.max_retries + 1):
            try:
                if "gpt-5-mini" in self.openrouter_model_name:
                    resp = self.client.chat.completions.create(
                        model=self.openrouter_model_name,
                        messages=messages,
                        temperature=temperature,
                        reasoning_effort="medium",
                        max_tokens=self.max_tokens,
                        **kwargs,
                    )
                else:
                    resp = self.client.chat.completions.create(
                        model=self.openrouter_model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        **kwargs,
                    )

                msg = resp.choices[0].message
                raw_content = getattr(msg, "content", None)
                content = raw_content.strip() if isinstance(raw_content, str) else ""

                if content == "":
                    raise ValueError("Empty response content from model")

                return content

            except Exception as e:
                last_err = e
                print(f"[ERROR] OpenAI SDK request failed attempt={attempt}: {repr(e)}")

                is_empty_content_err = (
                    isinstance(e, ValueError)
                    and "Empty response content" in str(e)
                )
                if attempt >= self.max_retries or (
                    not self._is_retryable(e) and not is_empty_content_err
                ):
                    break

                sleep = min(self.max_sleep, self.base_sleep * (2 ** attempt))
                sleep = sleep + random.uniform(0.0, 0.2 * sleep)
                time.sleep(sleep)

        print(f"[ERROR] OpenAI SDK final failure: {repr(last_err)}")
        return None

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float,
        **kwargs: Any,
    ):
        results = [None] * len(messages_list)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_map = {
                ex.submit(self.generate_one, msgs, temperature, **kwargs): i
                for i, msgs in enumerate(messages_list)
            }

            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="API threads"):
                i = fut_map[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    print(f"[ERROR] batch item failed: {e}")
                    results[i] = None

        return results