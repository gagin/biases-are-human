"""
Async model query runner for BDB.

Provides query_model and query_model_batch for sending multiple-choice
prompts to models via the OpenRouter API.
"""

import asyncio
import re
import string

import httpx

from bias_bench.env import get_api_key

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
CHOICE_LETTERS = string.ascii_uppercase  # A, B, C, ...


def _format_choices(choices: list[str]) -> str:
    """Format a list of choices as lettered options."""
    lines = []
    for i, choice in enumerate(choices):
        letter = CHOICE_LETTERS[i]
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def _extract_choice(text: str, valid_letters: set[str]) -> str | None:
    """
    Extract the first valid choice letter from a model response.
    Handles cases where the model adds explanation text.
    """
    # Strip whitespace
    text = text.strip()

    # First try: single letter (possibly followed by . or ) or space)
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in valid_letters:
        return match.group(1)

    # Second try: any uppercase letter in valid set
    for char in text.upper():
        if char in valid_letters:
            return char

    return None


async def query_model(
    model_id: str,
    prompt: str,
    choices: list[str],
    temperature: float = 0.0,
) -> dict:
    """
    Query a model with a multiple-choice prompt via OpenRouter.

    Args:
        model_id: OpenRouter model identifier (e.g. "openai/gpt-4o").
        prompt: The question or scenario text.
        choices: List of answer options (will be labeled A, B, C, ...).
        temperature: Sampling temperature (default 0.0 for deterministic).

    Returns:
        dict with keys:
            "choice"       - extracted letter (e.g. "A"), or None if unparseable
            "raw_response" - raw text from the model
            "model"        - model_id echoed back
            "usage"        - token usage dict from the API response
    """
    valid_letters = set(CHOICE_LETTERS[: len(choices)])
    formatted_choices = _format_choices(choices)

    system_message = (
        "You are taking a multiple-choice test. "
        "Respond with ONLY the single letter of your chosen answer (e.g. A, B, C). "
        "Do not include any explanation, punctuation, or additional text."
    )
    user_message = f"{prompt}\n\n{formatted_choices}"

    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
    }

    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    OPENROUTER_URL, headers=headers, json=payload
                )

                if response.status_code == 429 or response.status_code >= 500:
                    # Exponential backoff: 1s, 2s, 4s
                    wait = 2**attempt
                    await asyncio.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue

                response.raise_for_status()
                data = response.json()

                raw_response = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                choice = _extract_choice(raw_response, valid_letters)

                return {
                    "choice": choice,
                    "raw_response": raw_response,
                    "model": model_id,
                    "usage": usage,
                }

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                wait = 2**attempt
                await asyncio.sleep(wait)
                last_error = exc
                continue

    # All retries exhausted
    raise RuntimeError(
        f"query_model failed after {MAX_RETRIES} attempts for model {model_id!r}"
    ) from last_error


async def query_model_batch(
    model_id: str,
    prompts: list[dict],
    temperature: float = 0.0,
    max_concurrent: int = 3,
) -> list[dict]:
    """
    Run multiple model queries with semaphore-based concurrency limiting.

    Args:
        model_id: OpenRouter model identifier.
        prompts: List of dicts, each with keys "prompt" and "choices".
        temperature: Sampling temperature.
        max_concurrent: Maximum number of simultaneous in-flight requests.

    Returns:
        List of result dicts in the same order as the input prompts.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(item: dict) -> dict:
        async with semaphore:
            return await query_model(
                model_id=model_id,
                prompt=item["prompt"],
                choices=item["choices"],
                temperature=temperature,
            )

    tasks = [_run_one(item) for item in prompts]
    return list(await asyncio.gather(*tasks))
