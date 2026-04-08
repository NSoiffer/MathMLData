import time
import logging
import re
import argparse
import threading
from typing import Any, NamedTuple, Protocol, Callable, Iterator, cast

from compare_mathml_in_csv import setMathCATPreferences, areCanonicallyEqual, CanonicalResults
from dataclasses import dataclass
from enum import StrEnum
import xml.etree.ElementTree as ET
import yaml
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Conditional imports based on AI provider
try:
    from google import genai
    from google.genai import types
    from google.api_core import exceptions as google_exceptions
    GEMINI_IMPORT_ERROR = None
except ImportError:
    GEMINI_IMPORT_ERROR = ImportError(
        "Google GenAI library not available. Install with: pip install google-genai"
    )
try:
    from openai import OpenAI
    from openai import APIError, RateLimitError, APIConnectionError
    OPENAI_IMPORT_ERROR = None
except ImportError:
    OPENAI_IMPORT_ERROR = ImportError(
        "OpenAI library not available. Install with: pip install openai"
    )

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

BRAILLE_REGEX = re.compile(r'([\u2800-\u28ff]+)')
MATHML_REGEX: re.Pattern[str] = re.compile(r'<math.*?</math>')
# ============================================================
# this comes from alt_use_ai.py
# ============================================================


class BrailleCode(StrEnum):
    NEMETH = "nemeth"
    UEB = "ueb"


@dataclass(slots=True)
class MathExample:
    mathml: str
    braille: str
    code: BrailleCode


@dataclass(slots=True)
class SymbolMapping:
    char: str
    nemeth: str | None = None
    nemeth_numeric: str | None = None
    ueb: str | None = None
    reverse_hint: str | None = None


class RunConfig(NamedTuple):
    """Configuration values for an AI API run."""
    braille_code: BrailleCode
    gen_braille: bool
    ai_provider: str
    model: str
    service_tier: str
    apiKeyName: str
    batch_size: int
    start_index: int
    n_examples: int
    symbol_mappings: list[SymbolMapping]
    example_braille_file: str
    example_mathml_file: str
    input_braille_file: str
    input_mathml_file: str

    def print_config(self, n_tests: int | None = None, short: bool = False) -> str:
        """Return configuration values as a formatted string."""
        lines = []
        lines.append("\nConfiguration:")
        lines.append(f"  Braille Code: {self.braille_code}")
        lines.append(f"  Generate Braille: {self.gen_braille}")
        lines.append(f"  Model: {self.model} {'(' + self.service_tier + ')' if self.ai_provider == 'openai' else ''}")
        lines.append(f"  API Key: {self.apiKeyName}")
        lines.append(f"  Batch Size: {self.batch_size}")
        lines.append(f"  Number of Examples: {self.n_examples}")
        if n_tests is not None:
            lines.append(f"  Number of Tests: {n_tests} starting at {self.start_index}")
        lines.append(f"  Example Braille File: {self.example_braille_file}")
        lines.append(f"  Example MathML File: {self.example_mathml_file}")
        lines.append(f"  Input Braille Dir: {self.input_braille_file}")
        lines.append(f"  Input MathML Dir: {self.input_mathml_file}")
        return "\n".join(lines)


# ============================================================
# Prompt Builders (Unified)
# ============================================================

def build_instructions(
    config: RunConfig,
    symbol_block: str,
) -> str:

    if config.gen_braille:
        header = (
            f"You are an expert MathML to {config.braille_code.value.upper()} Braille translator.\n"
        )
        example_prolog = (
            "Use the following pairs of examples to infer the correct mapping from MathML to"
            f" {config.braille_code.value.upper()} braille.\n\n"
        )
        test_prolog = (
            f"After the examples, translate the following MathML expressions into "
            f"{config.braille_code.value.upper()} Braille. "
            "Each MathML is numbered and on its own line\n"
            "Return ONLY Unicode braille characters. "
            "It is important to pay attention to generating Unicode braille spaces when needed in the braille.\n"
            "Relational operators such as <, >, ≤, ≥, =, ≠ almost always need to have spaces on the left and right "
            "unless they are in a script position.\n"

        )
        if config.braille_code == BrailleCode.NEMETH:
            test_prolog += (
                "Some things to remember about Nemeth Braille: \n"
                "- the number sign indicator ⠼ that precedes digits is ONLY needed in these two cases: \n"
                "  1. at the start of a line, after a space, or after punctuation. \n"
                "  2. if a digit follows a minus sign that is at the start of a line, after a space, or after "
                "punctuation.\n"
                "- the English letter indicator ⠰ that precedes Roman letters is ONLY needed in these five cases:\n"
                "  1. after a space;\n"
                "  2. before a single lowercase English letter when it is isolated or followed only by punctuation, "
                'potentially with intervening open or close characters;\n'
                "  3. if the MathML consists of only an 'mi' or 'mtext' element, "
                "and that element contains a single letter;\n"
                "  4. if the letter is in bold, italic, or some other non-Roman style;\n"
                "  5. if the lowercase letter is part of a Roman numeral.\n"
                "- the English letter indicator ⠰ is NEVER needed between two Roman letters, "
                "following a function name, before an operator, or after an operator.\n"
                "- a letter with an integer subscript should NOT use a subscript indicator if the subscript is not "
                "inside of a subscript or superscript.\n"
                "- the braille should never start or end with a braille space."
            )
        else:  # UEB
            test_prolog += (
                "Some things to remember about UEB Braille:\n"
                "- grade 1 symbol indicators ⠰, grade 1 word indicators ⠰⠰, and grade 1 passage indicators ⠰⠰⠰ are "
                "often needed at the start of a line.\n"
                "- in general, you want to minimize the use of grade 1 symbol indicators. "
                "Use them only when they result in shorter braille than using grade 1 word or passage indicators. "
                "If there are more than two braille spaces (⠀), and a grade 1 indicator is needed "
                "in the first three braille characters, use the grade 1 passage indicators.\n"
                "- A grade 1 indicator only sets grade 1 mode for the next symbol and is not needed "
                "before the letters 'a', 'i', and 'o'.\n"
                "- A grade 1 word indicator only sets grade 1 mode until the next space.\n"
                "- A number sign indicator ⠼ sets grade 1 word mode.\n"
                "- A letter or unbroken sequence of letters is 'standing alone' if the symbols before and after the "
                "letter or sequence are spaces, hyphens, dashes, or any combination thereof, including some common "
                "punctuation. An opening bracketing character before a sequence or closing bracketing character after "
                "a sequence should be included in the above definition of 'standing alone'. A single letter (excluding "
                "a, i and o) is considered 'standing alone' if it is preceded by a space.\n"
                "- A grade 1 indicator ⠰ is needed before a standing alone letter or sequence of letters.\n"
                "- the number sign indicator ⠼ is ONLY needed before digits and starts numeric mode.\n"
                "- All fraction, root, subscript, superscript, etc., start, middle, and end indicators MUST "
                "be in grade 1 mode; a grade 1 indicator is required before the fraction, etc., indicator if it is not "
                " already in grade 1 mode.\n"
                "- Numeric mode includes the digits and the fraction line (⠌) for simple numeric fractions. It also "
                "includes ',', '.', and spaces when they appear inside of MathML mn elements.\n"
                "- if the lowercase letters a-j follow a digit, you MUST use a grade 1 indicator ⠰ before the letter.\n"
                "- numeric fraction do not use start or end fraction indicators, but all other fractions start with ⠷, "
                "end with ⠾, and use ⠨⠌ as the fraction bar.\n"
                "- all subscripts MUST start with the subscript indicator ⠢, and must be in grade 1 mode.\n"
            )
        symbols_text = (
            "Here is a reminder of the mapping of some Unicode characters to their "
            f"representation in {config.braille_code.value.upper()} braille that you may need to use:\n"
            f"{symbol_block}"
        )
        test_prolog += (
            "Do NOT include the input line numbers (e.g., '1)').\n"
            "Add '|next-item|' between each braille output.\n"
        )

    else:
        header = (
            f"You are a expert {config.braille_code.value.upper()} braille to MathML translator.\n"
            "Use the following pairs of examples to infer the correct mapping from"
            f" {config.braille_code.value.upper()} braille to MathML.\n\n"
        )
        test_prolog = (
            f"Now translate the following {config.braille_code.value.upper()} Braille into MathML.\n"
            "Return ONLY valid MathML markup. MathML must start with a <math> tag and end with a </math> tag. "
            "Do NOT include the input line numbers (e.g., '1)', '2)', etc.).\n"
            "Add '|next-item|' between each MathML output."
        )

        symbols_text = (
            f"Here is a reminder of the mapping of some Unicode {config.braille_code.value.upper()}"
            f"braille characters and how they map to Unicode non-braille characters"
            f"that you may need to use:\n{symbol_block}\n\n"
        )
        example_prolog = (
            "Use the following pairs of examples to infer the correct mapping from "
            f" {config.braille_code.value.upper()} braille to MathML.\n\n"
        )

    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(f"Instructions: {header + symbols_text + example_prolog}\n{test_prolog}\n\n\n")

    return header + symbols_text + example_prolog + test_prolog


class UsageMetadata(Protocol):
    """Protocol for usage metadata objects."""
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


class OpenAIUsageMetadata:
    """Simple class to hold token usage information for GPT."""
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.total_token_count = total_tokens


class AIClient(Protocol):
    """Protocol for AI clients."""
    pass


def create_gemini_client(api_key: str) -> genai.Client:
    """Create and return a Gemini client."""
    return genai.Client(api_key=api_key, http_options={"timeout": 2400000})


def create_openai_client(api_key: str) -> OpenAI:
    """Create and return a GPT client."""
    return OpenAI(api_key=api_key, timeout=2400.0)


def _generate_with_retry_common(
    client: Any,
    config: RunConfig,
    examples: list[dict[str, Any]],
    tests: list[str],
    symbol_block: str,
    gemini_cache_id: str,
    max_retries: int,
    depth: int,
    create_stream_func: Callable[[genai.Client, RunConfig, list[dict[str, Any]], str, str, str], Iterator[Any]],
    process_chunk_func: Callable[[genai.Client, list[str]], tuple[str | None, Any | None, str | None] | None],
    get_fallback_usage_func: Callable[
        [Any, RunConfig, list[dict[str, Any]], str],
        tuple[Any | None, str | None] | None
    ] | None,
    is_max_tokens_func: Callable[[str | None], bool],
    is_success_finish_func: Callable[[str | None], bool],
    handle_retry_exception_func: Callable[[Exception, int, int, str, str], tuple[bool, str | None]],
    sum_usage_func: Callable[[Any, Any], Any],
    default_error_usage: Any,
    recursive_call_func: Callable[
        [Any, RunConfig, list[dict[str, Any]], list[str], str, str, int, int],
        tuple[str, Any, float]
    ]
) -> tuple[str, Any, float]:
    """Common retry logic shared between Gemini and GPT."""
    indent = "  " * depth
    t0 = time.perf_counter()
    time_to_first_token = -1000.0
    delay = 30
    full_text_list: list[str] = []
    final_usage: Any = default_error_usage
    run_info = f"{'to-' if config.gen_braille else 'from-'}{config.braille_code}"

    for attempt in range(1, max_retries + 1):
        try:
            # Add line numbers to content
            numbered_tests = "\n".join(f"{i}) {s}" for i, s in enumerate(tests, 1))
            response_stream = create_stream_func(
                client, config, examples, numbered_tests, symbol_block, gemini_cache_id
            )

            full_text_list = []
            first_token_received = False
            final_usage = default_error_usage
            finish_reason = None

            for chunk in response_stream:
                if not first_token_received:
                    time_to_first_token = time.perf_counter() - t0
                    print(f"⚡ Time to First Token for {run_info}: {time_to_first_token:.2f} seconds")
                    first_token_received = True

                chunk_result = process_chunk_func(chunk, full_text_list)
                if chunk_result:
                    text, usage, reason = chunk_result
                    if text:
                        full_text_list.append(text)
                    if usage is not None:
                        final_usage = usage
                    if reason is not None:
                        finish_reason = reason

            # Try to get usage from fallback if not available (GPT only)
            if (
                get_fallback_usage_func
                and hasattr(final_usage, 'total_token_count')
                and final_usage.total_token_count == 0
            ):
                fallback_result = get_fallback_usage_func(
                    client,
                    config,
                    examples,
                    numbered_tests
                )
                if fallback_result:
                    fallback_usage, fallback_reason = fallback_result
                    if fallback_usage is not None:
                        final_usage = fallback_usage
                    if fallback_reason is not None:
                        finish_reason = fallback_reason

            if is_max_tokens_func(finish_reason):
                raise ValueError("MAX_TOKENS")

            if not is_success_finish_func(finish_reason):
                raise Exception(f"Incomplete generation: {finish_reason}")

            print(f"\n\n--- Performance Summary for {run_info} ---")
            total_time = time.perf_counter() - t0
            print(f"Total Latency:    {total_time:.2f} s")
            print(f"Time to 1st Token:{time_to_first_token:.2f} s")
            print(f"Generation Time:  {total_time - time_to_first_token:.2f} s (Streaming duration)")
            return "".join(full_text_list), final_usage, total_time - time_to_first_token

        except ValueError as e:
            if str(e) != "MAX_TOKENS":
                raise e

            print(f"{indent}[!] In {run_info}: MAX_TOKENS hit on {len(tests)} lines.")

            if len(tests) <= 1:
                print(f"{indent}[X] Critical in {run_info}: Single input line is too large.")
                raise e

            mid = len(tests) // 2
            left_part = tests[:mid]
            right_part = tests[mid:]

            print(f"{indent}    -> Splitting: {len(left_part)} lines | {len(right_part)} lines")

            text_a, usage_a, _ = recursive_call_func(
                client, config, examples, left_part, symbol_block, gemini_cache_id, max_retries, depth + 1
            )
            text_b, usage_b, _ = recursive_call_func(
                client, config, examples, right_part, symbol_block, gemini_cache_id, max_retries, depth + 1
            )

            if text_a is None or text_b is None:
                return "Error", default_error_usage, time.perf_counter() - t0 - time_to_first_token

            return (text_a + '|next-item|' + text_b,
                    sum_usage_func(usage_a, usage_b),
                    time.perf_counter() - t0 - time_to_first_token
                    )

        except Exception as e:
            should_retry, retry_msg = handle_retry_exception_func(e, attempt, max_retries, run_info, indent)
            if should_retry:
                print(f"{indent}{retry_msg} Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"{indent}Exception Type: {type(e).__name__}")
                print(f"{indent}[X] Critical Error in {run_info}: {e}")
                if len(full_text_list) > 0:
                    total_time = time.perf_counter() - t0
                    return "".join(full_text_list), final_usage, total_time - time_to_first_token
                raise e

    print(f"{indent}[X] Failed after max retries in {run_info}.")
    if len(full_text_list) > 0:
        total_time = time.perf_counter() - t0
        return "".join(full_text_list), final_usage, total_time - time_to_first_token
    else:
        return "Error", default_error_usage, time.perf_counter() - t0 - time_to_first_token


def generate_with_retry_gemini(
    client: genai.Client,
    config: RunConfig,
    examples: list[dict[str, Any]],
    tests: list[str],
    symbol_block: str,
    gemini_cache_id: str,
    max_retries: int = 3,
    depth: int = 0
) -> tuple[str, Any, float]:
    """Generate with retry logic for Gemini API."""
    def create_stream(
        client: genai.Client,
        config: RunConfig,
        examples: list[dict[str, Any]],
        numbered_tests: str,
        symbol_block: str,
        gemini_cache_id: str
    ) -> Iterator[Any]:
        test_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=numbered_tests)]
        )

        is_gemma = "gemma" in config.model.lower()
        gemini_config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                )
            ],
            temperature=0.0,
            max_output_tokens=4096 if is_gemma else None,
        )
        if gemini_cache_id:
            gemini_config.cached_content = gemini_cache_id
        else:
            gemini_config.system_instruction = build_instructions(config, symbol_block)
        return client.models.generate_content_stream(
            model=config.model,
            config=gemini_config,
            contents=[test_content] if gemini_cache_id else examples + [test_content],
        )

    def process_chunk(chunk: Any, full_text_list: list[str]) -> tuple[str | None, Any | None, str | None] | None:
        text = chunk.text if hasattr(chunk, 'text') else None
        usage = chunk.usage_metadata if hasattr(chunk, 'usage_metadata') else None
        reason = chunk.candidates[0].finish_reason if (hasattr(chunk, 'candidates') and chunk.candidates) else None
        return (text, usage, reason)

    def is_max_tokens(finish_reason: str | None) -> bool:
        return finish_reason == "MAX_TOKENS"

    def is_success_finish(finish_reason: str | None) -> bool:
        return finish_reason == "STOP"

    def handle_retry_exception(
        e: Exception,
        attempt: int,
        max_retries: int,
        run_info: str,
        indent: str
    ) -> tuple[bool, str | None]:
        err = str(e)
        if "503" in err or "UNAVAILABLE" in err:
            return True, f"[!] 503 Unavailable (Attempt {attempt}/{max_retries}) {run_info}."
        if "499" in err or "CANCELLED" in err:
            return True, f"[!] 499 Cancelled (Attempt {attempt}/{max_retries}) {run_info}."
        return False, None

    return _generate_with_retry_common(
        client=client,
        config=config,
        examples=examples,
        tests=tests,
        symbol_block=symbol_block,
        gemini_cache_id=gemini_cache_id,
        max_retries=max_retries,
        depth=depth,
        create_stream_func=create_stream,
        process_chunk_func=process_chunk,
        get_fallback_usage_func=None,
        is_max_tokens_func=is_max_tokens,
        is_success_finish_func=is_success_finish,
        handle_retry_exception_func=handle_retry_exception,
        sum_usage_func=_sum_usage_gemini,
        default_error_usage=None,
        recursive_call_func=generate_with_retry_gemini
    )


def generate_with_retry_openai(
    client: OpenAI,
    config: RunConfig,
    examples: list[dict[str, Any]],
    tests: list[str],
    symbol_block: str,
    gemini_cache_id: str,
    max_retries: int = 3,
    depth: int = 0
) -> tuple[str, OpenAIUsageMetadata, float]:
    """Generate with retry logic for "OpenAI" API."""
    messages_cache: list[dict[str, str]] | None = None

    def create_stream(
        client: OpenAI,
        config: RunConfig,
        examples: list[dict[str, Any]],
        numbered_tests: str,
        symbol_block: str,
        gemini_cache_id: str
    ) -> Iterator[Any]:
        nonlocal messages_cache
        # Build messages array: system message, then examples (if list), then user/assistant for payload
        messages_cache = [{"role": "system", "content": build_instructions(config, symbol_block)}]
        # Insert example messages into the array
        messages_cache.extend(examples)
        messages_cache.extend([
            {"role": "user", "content": numbered_tests}
        ])
        return call_openai_model(
            client=client,
            config=config,
            messages=messages_cache,
            stream=True,
        )

    def process_chunk(
        chunk: Any, full_text_list: list[str]
    ) -> tuple[str | None, OpenAIUsageMetadata | None, str | None] | None:
        text: str | None = None
        usage: OpenAIUsageMetadata | None = None
        reason: str | None = None

        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                text = delta.content

            if chunk.choices[0].finish_reason:
                reason = chunk.choices[0].finish_reason

        if chunk.usage:
            usage = OpenAIUsageMetadata(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0
            )

        return (text, usage, reason)

    def get_fallback_usage(
        client: OpenAI,
        config: RunConfig,
        examples: list[dict[str, Any]],
        payload_text: str
    ) -> tuple[OpenAIUsageMetadata | None, str | None] | None:
        try:
            if messages_cache is None:
                return None
            response = call_openai_model(
                client=client,
                config=config,
                messages=messages_cache,
                stream=False
            )
            usage = None
            reason = None
            if response.usage:
                usage = OpenAIUsageMetadata(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0
                )
            if response.choices and len(response.choices) > 0:
                reason = response.choices[0].finish_reason
            return (usage, reason)
        except Exception:
            return (None, None)

    def is_max_tokens(finish_reason: str | None) -> bool:
        return finish_reason == "length"

    def is_success_finish(finish_reason: str | None) -> bool:
        return finish_reason == "stop"

    def handle_retry_exception(
        e: Exception,
        attempt: int,
        max_retries: int,
        run_info: str,
        indent: str
    ) -> tuple[bool, str | None]:
        error_str = str(e)
        if "rate_limit" in error_str.lower() or "429" in error_str or isinstance(e, RateLimitError):
            return True, f"[!] Rate Limit Error (Attempt {attempt}/{max_retries}) {run_info}."
        elif "connection" in error_str.lower() or isinstance(e, APIConnectionError):
            return True, f"[!] Connection Error (Attempt {attempt}/{max_retries}) {run_info}."
        return False, None

    return _generate_with_retry_common(
        client=client,
        config=config,
        examples=examples,
        tests=tests,
        symbol_block=symbol_block,
        gemini_cache_id=gemini_cache_id,
        max_retries=max_retries,
        depth=depth,
        create_stream_func=create_stream,
        process_chunk_func=process_chunk,
        get_fallback_usage_func=get_fallback_usage,
        is_max_tokens_func=is_max_tokens,
        is_success_finish_func=is_success_finish,
        handle_retry_exception_func=handle_retry_exception,
        sum_usage_func=_sum_usage_openai,
        default_error_usage=OpenAIUsageMetadata(),
        recursive_call_func=generate_with_retry_openai
    )


def _sum_usage_gemini(usage1: Any, usage2: Any) -> Any:
    """Helper to sum two Gemini UsageMetadata objects."""
    if not usage1:
        return usage2
    if not usage2:
        return usage1

    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage1.prompt_token_count + usage2.prompt_token_count,
        candidates_token_count=usage1.candidates_token_count + usage2.candidates_token_count,
        total_token_count=usage1.total_token_count + usage2.total_token_count
    )


def _sum_usage_openai(usage1: OpenAIUsageMetadata, usage2: OpenAIUsageMetadata) -> OpenAIUsageMetadata:
    """Helper to sum two GPT UsageMetadata objects."""
    if not usage1:
        return usage2
    if not usage2:
        return usage1

    return OpenAIUsageMetadata(
        prompt_tokens=usage1.prompt_token_count + usage2.prompt_token_count,
        completion_tokens=usage1.candidates_token_count + usage2.candidates_token_count,
        total_tokens=usage1.total_token_count + usage2.total_token_count
    )


def call_openai_model(
    client: OpenAI,
    config: RunConfig,
    messages: list[dict[str, str]],
    stream: bool = True,
) -> Any:
    """
    Call OpenAI model with the given messages.

    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: List of message dicts with 'role' and 'content'
        stream: Whether to stream the response

    Returns:
        If stream=True: Returns a stream object that can be iterated
        If stream=False: Returns the full response object
    """
    params: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": stream,
        "service_tier": config.service_tier,
        "temperature": 0.1,

    }
    return client.chat.completions.create(**params)


def get_context_cache_id(
    client: genai.Client,
    config: RunConfig,
    symbol_block: str,
    examples: list[dict[str, Any]]
) -> types.CachedContent:
    """Get the cache id for the context of the examples and symbol block."""
    cache_id = client.caches.create(
        model=config.model,
        config=types.CreateCachedContentConfig(
            display_name='braille_translation_examples',
            system_instruction=build_instructions(config, symbol_block),
            contents=examples,
            ttl="3600s",
        )
    )
    time.sleep(15)   # wait for the cache to be created -- was getting 503 and this is a suggested workaround
    return cache_id


def convert_input_with_model(
    config: RunConfig,
    examples: list[dict[str, Any]],
    tests: list[str],
) -> tuple[list[str], dict[str, int], float]:
    """
    Returns
    Splits input into batches, processes them with retries/streaming,
    tracks token usage, and measures pure generation time.
    """
    ai_provider = config.ai_provider.lower()
    run_info = f"{'to-' if config.gen_braille else 'from-'}{config.braille_code}"

    # Setup Client
    api_key = os.environ.get(config.apiKeyName)
    if not api_key:
        raise ValueError(f"Please set the {config.apiKeyName} environment variable.")

    # if ai_provider == "gemini":
    #     client = create_gemini_client(api_key)
    #     generate_func = generate_with_retry_gemini
    #     retry_exceptions = (google_exceptions.ServiceUnavailable, google_exceptions.ServerError)
    # elif ai_provider == "openai":
    #     client = create_openai_client(api_key)
    #     generate_func = generate_with_retry_openai
    #     retry_exceptions = (APIError, RateLimitError, APIConnectionError)
    # else:
    #     raise ValueError(f"Unknown AI provider: {ai_provider}. Must be 'gemini' or 'openai'")

    # 1. Initialize accumulators
    all_results = ""
    total_tokens: dict[str, int] = {"prompt": 0, "candidates": 0, "total": 0}
    total_generation_time: float = 0.0
    first_attempt = True

    # because we might batch the instructions, we need to extract the symbols from all the tests
    paid_tier = False   # TODO: apparently fails if this is an unpaid teir
    if paid_tier and config.ai_provider == "gemini" and len(examples) > 300:
        used_symbols = set().union(*(extract_symbols_from_mathml(test) for test in tests))
        batch_symbol_block = build_symbol_block(
            used_symbols,
            config.symbol_mappings,
            config.braille_code,
            config.gen_braille
        )
        cached_content = get_context_cache_id(cast(genai.Client, client), config, batch_symbol_block, examples)
        gemini_cache_id: str = cached_content.name if cached_content and cached_content.name else ""
    else:
        gemini_cache_id = ""
    # 2. Loop through the data in chunks
    for i in range(0, len(tests), config.batch_size):
        # I thought we could do this once at the beginning, but it seems to be necessary to reestablish the connection when there are lots of batches
        if ai_provider == "gemini":
            client = create_gemini_client(api_key)
            generate_func = generate_with_retry_gemini
            retry_exceptions = (google_exceptions.ServiceUnavailable, google_exceptions.ServerError)
        elif ai_provider == "openai":
            client = create_openai_client(api_key)
            generate_func = generate_with_retry_openai
            retry_exceptions = (APIError, RateLimitError, APIConnectionError)
        else:
            raise ValueError(f"Unknown AI provider: {ai_provider}. Must be 'gemini' or 'openai'")

        batch = tests[i:i + config.batch_size]
        batch_id = (i // config.batch_size) + 1
        print(
            f"\n--- Processing Batch {batch_id} "
            f"(Items {i+1} to {i+len(batch)}) "
            f"{'to-' if config.gen_braille else 'from-'}{config.braille_code} ---"
        )

        if gemini_cache_id == "":
            if config.gen_braille:
                used_symbols = set().union(*(extract_symbols_from_mathml(test) for test in batch))
            else:
                used_symbols = set().union(*(extract_symbols_from_braille(test) for test in batch))
            symbol_block = build_symbol_block(
                used_symbols,
                config.symbol_mappings,
                config.braille_code,
                config.gen_braille
            )
        else:
            symbol_block = ""

        # 3. Call helper (now returns duration too)
        try:
            batch_text, batch_usage, batch_time = generate_func(
                cast(Any, client),
                config,
                examples,
                batch,
                symbol_block,
                gemini_cache_id,
                3
            )
        except retry_exceptions as e:
            if first_attempt:
                # reestablish connection and try one more time
                if ai_provider == "gemini":
                    client = create_gemini_client(api_key)
                else:
                    client = create_openai_client(api_key)
                first_attempt = False
                try:
                    batch_text, batch_usage, batch_time = generate_func(
                        cast(Any, client),
                        config,
                        examples,
                        batch,
                        symbol_block,
                        gemini_cache_id,
                        3
                    )
                except Exception as e:
                    print(f"Exception raised during retry: {e}")
                    break
            else:
                print(f"Exception raised twice during generation: {e}")
                batch_text, batch_usage, batch_time = None, None, 0.0
                break

        except Exception as e:
            print(f"Exception raised during generation: {e}")
            batch_text, batch_usage, batch_time = None, None, 0.0
            break

        # 4. Process results
        if batch_text:
            all_results += '|next-item|' + batch_text

        # 5. Update stats
        if batch_usage:
            total_tokens["prompt"] += batch_usage.prompt_token_count
            total_tokens["candidates"] += batch_usage.candidates_token_count
            total_tokens["total"] += batch_usage.total_token_count

        if batch_time:
            total_generation_time += batch_time
            print(f"   > Batch Time: {batch_time:.2f}s")
            if batch_usage:
                print(f"   > Batch Token Usage ({run_info}): {batch_usage.total_token_count} "
                      f"(Prompt: {batch_usage.prompt_token_count}, Output: {batch_usage.candidates_token_count})")

    # delete the cache
    if gemini_cache_id:
        cast(genai.Client, client).caches.delete(name=gemini_cache_id)

    # look at tests to see if we are generating MathML or braille
    # trim the start and end, then split the string at '|next-item|' and return a list of strings
    text = all_results
    if config.gen_braille:
        matches = list(BRAILLE_REGEX.finditer(text))
        if not matches:
            print(f"\n\n[!] Could not find braille chars in the response\n: '{text}'\n\n")
            return [], total_tokens, total_generation_time
        i_start = matches[0].start()
        i_end = matches[-1].end()
        regex = BRAILLE_REGEX
    else:
        i_start = text.find("<math")
        i_end = text.rfind("</math>") + len("</math>")
        if i_start == -1 or i_end == -1:
            print(f"\n\n[!] Could not find braille chars in the response\n: '{text}'\n\n")
        regex = MATHML_REGEX
    # Return the substring including everything between the first and last Braille char/MathML start/end tags
    as_list = []
    for item in text[i_start:i_end].split("|next-item|"):
        match = regex.search(item)
        if match:
            as_list.append(match.group(0).strip())
        else:
            as_list.append(item.strip())

    return as_list, total_tokens, total_generation_time


# ============================================================
# Symbol Extraction + Structural Context
# ============================================================
def extract_symbols_from_mathml(mathml: str) -> set[str]:
    root = ET.fromstring(mathml)
    symbols: set[str] = set()
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in {"mo", "mi", "mn", "mtext"} and elem.text:
            # if elem.text and not elem.text.strip().isalnum():
            symbols.update(elem.text)
    return symbols


def extract_symbols_from_braille(braille: str) -> set[str]:
    return set(braille) if braille else set()


def extract_structural_context(mathml: str) -> list[str]:
    root = ET.fromstring(mathml)
    notes: list[str] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "mn" and elem.text:
            if "," in elem.text or "." in elem.text:
                notes.append("Comma inside <mn> → numeric comma.")
        if tag == "mo" and elem.text:
            if elem.text == "," or elem.text == ".":
                notes.append("Comma as <mo> → argument separator.")
    return sorted(set(notes))


def count_complicated_elements(mathml: str) -> int:
    root = ET.fromstring(mathml)
    count = 0
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in {"msub", "msup", "msubsup", "mmultiscripts",
                   "msqrt", "mroot",
                   "mfrac", "menclose",
                   "munder", "mover", "munderover",
                   "mtr", "mlabeledtr", "mtd", "mtable"}:
            count += 1
    return count


# ============================================================
# Symbol Mapping Blocks
# ============================================================
def _extract_first_t(obj: Any) -> str | None:
    if isinstance(obj, dict):
        if "t" in obj:
            return obj["t"]

        if "test" in obj:
            tb: dict[str, Any] = obj["test"]

            for key in ("then", "then_test"):
                if key in tb:
                    for entry in tb[key]:
                        tval = _extract_first_t(entry)
                        if tval:
                            return tval

            for key in ("else", "else_test"):
                if key in tb:
                    for entry in tb[key]:
                        tval = _extract_first_t(entry)
                        if tval:
                            return tval

    elif isinstance(obj, list):
        for entry in obj:
            tval = _extract_first_t(entry)
            if tval:
                return tval

    return None


def _load_yaml_mapping_file_simple(path: str) -> dict[str, str]:
    """
    Loads a YAML file where the top-level structure is a LIST.
    Each list item may be:
      A) {char: "(", rules: [...]}
      B) {"(": [ ...rule objects... ]}
    Returns:
        { char: first_braille_t }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []

    result: dict[str, str] = {}

    for entry in raw:
        # ----------------------------
        # Case A: explicit fields
        # ----------------------------
        if isinstance(entry, dict) and "char" in entry:
            char = entry["char"]
            rules = entry.get("rules", [])
            tval = _extract_first_t(rules)
            if tval:
                result[char] = tval
            continue

        # ----------------------------
        # Case B: {"(": [ ... ]}
        # ----------------------------
        if isinstance(entry, dict) and len(entry) == 1:
            char, rules = next(iter(entry.items()))
            tval = _extract_first_t(rules)
            if tval:
                result[char] = tval
            continue

        raise ValueError(f"Unrecognized YAML mapping entry: {entry}")

    return result


def load_symbol_mappings(nemeth_path: str, ueb_path: str) -> list[SymbolMapping]:
    nemeth_map = _load_yaml_mapping_file_simple(nemeth_path)
    ueb_map = _load_yaml_mapping_file_simple(ueb_path)

    merged: dict[str, SymbolMapping] = {}

    # Nemeth first
    for ch, braille in nemeth_map.items():
        merged[ch] = SymbolMapping(
            char=ch,
            nemeth=braille,
            nemeth_numeric=None,
            ueb=None
        )

    # UEB second
    for ch, braille in ueb_map.items():
        if ch in merged:
            merged[ch].ueb = braille
        else:
            merged[ch] = SymbolMapping(
                char=ch,
                nemeth=None,
                nemeth_numeric=None,
                ueb=braille
            )

    return list(merged.values())


def filter_symbol_mappings(mappings: list[SymbolMapping], used: set[str]) -> list[SymbolMapping]:
    return [m for m in mappings if m.char in used]


def build_symbol_reference_block(mappings: list[SymbolMapping], code: BrailleCode) -> str:
    if not mappings:
        return ""
    lines = []
    if code is BrailleCode.NEMETH:
        lines.append("Symbol reference (Nemeth):")
        for m in mappings:
            if m.nemeth or m.nemeth_numeric:
                base = m.nemeth or ""
                numeric = f" (numeric: {m.nemeth_numeric})" if m.nemeth_numeric else ""
                lines.append(f"  '{m.char}' → {base}{numeric}")
    else:
        lines.append("Symbol reference (UEB):")
        for m in mappings:
            if m.ueb:
                lines.append(f"  '{m.char}' → {m.ueb}")
    return "\n".join(lines) + "\n\n"


def build_reverse_symbol_reference_block(mappings: list[SymbolMapping], code: BrailleCode) -> str:
    if not mappings:
        return ""
    lines = []
    if code is BrailleCode.NEMETH:
        lines.append("Braille→symbol reference (Nemeth):")
        for m in mappings:
            if m.nemeth:
                lines.append(f"  {m.nemeth} → '{m.char}'")
            if m.nemeth_numeric:
                lines.append(f"  {m.nemeth_numeric} (numeric) → '{m.char}'")
    else:
        lines.append("Braille→symbol reference (UEB):")
        for m in mappings:
            if m.ueb:
                lines.append(f"  {m.ueb} → '{m.char}'")
    return "\n".join(lines) + "\n\n"


def build_symbol_block(
    used_symbols: set[str],
    mappings: list[SymbolMapping],
    code: BrailleCode,
    generate_braille: bool
) -> str:
    if generate_braille:
        return build_symbol_reference_block(
            filter_symbol_mappings(mappings, used_symbols),
            code
        )
    else:
        used = []
        for m in mappings:
            cells = []
            if code is BrailleCode.NEMETH:
                if m.nemeth:
                    cells.append(m.nemeth)
                if m.nemeth_numeric:
                    cells.append(m.nemeth_numeric)
            else:
                if m.ueb:
                    cells.append(m.ueb)
            if any(c and any(ch in used_symbols for ch in c) for c in cells):
                used.append(m)
        return build_reverse_symbol_reference_block(used, code)

# ------------------------------------------------------------


def write_results_to_file(input: list[str],
                          computed_output: list[str],
                          expected_output: list[str],
                          info: dict[str, int],  # time is in ms
                          output_file: str,
                          config: RunConfig | None = None) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    """
    usage_info = str(info)[1:-1].replace("'", "").replace(": ", "=")
    print(f"Generated {len(computed_output)} outputs. Stats: {usage_info}ms")
    is_mathml_output = expected_output[0].startswith('<math')
    if not isinstance(computed_output, list):
        print(f"Error: Computed output is a {type(computed_output)},\
              {len(computed_output) if isinstance(computed_output, list) else 0} items")
        return
    if is_mathml_output and not computed_output[0].startswith('<math'):
        print("Computed output does not appear to be MathML--first 5 lines:\n", computed_output[:5])
        return
    if not is_mathml_output and not BRAILLE_REGEX.match(computed_output[0][0]):
        print("Computed output does not appear to be MathML--last 5 lines:\n", computed_output[len(computed_output)-5:])
        return

    # initial MathCAT
    setMathCATPreferences({})

    with open(output_file, "w", encoding="utf-8") as f:
        # Write variable values from main() at the start
        if config:
            # Write config to file with # prefix on each line
            config_str = config.print_config()
            for line in config_str.split('\n'):
                if line.strip():  # Skip empty lines
                    f.write(f"# {line}\n")
            f.write("#\n")

        match_count = 0
        bad_mathml_count = 0
        f.write(f"# {len(computed_output)} items. "
                f"Usage info: {usage_info}ms, "
                f"TPS={(1000 * info['time']/info['candidates'])}.2f\n#\n")
        if is_mathml_output:
            f.write("\n# NOT Normalized MathML\n")
        f.write("# Match | Test Input | Expected | Computed\n")
        for tests, computed, expected in zip(input, computed_output, expected_output):
            try:
                if is_mathml_output:
                    checked = areCanonicallyEqual(expected, computed)
                else:
                    checked = CanonicalResults(expected.strip() == computed.strip(), "", "")
                if checked.isEqual:
                    match_count += 1
            except Exception:
                checked = CanonicalResults(False, "", "")
            match = "✓" if checked.isEqual else "✗"
            f.write(f"{match} | {tests} | {expected} | {computed}\n")
        if is_mathml_output:
            f.write("\n#===========\n")
            f.write("\n# Normalized MathML\n")
            f.write("Match | Test Input | Expected | Computed\n")
            for tests, computed, expected in zip(input, computed_output, expected_output):
                try:
                    checked = areCanonicallyEqual(expected, computed)
                except Exception as e:
                    print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                    checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)
                    bad_mathml_count += 1
                match = "✓" if checked.isEqual else "✗"
                f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        f.write(
            f"# Matches: {match_count} out of {len(computed_output)}: "
            f"{((match_count/len(computed_output))*100):.0f}%."
        )
        print(f"Matches: {match_count} out of {len(computed_output)}: {(match_count/len(computed_output)*100):.0f}%. "
              f"Bad MathML: {bad_mathml_count} ({(bad_mathml_count/len(computed_output)*100):.0f}%).\n"
              f"Results written to {output_file}. ")


def readMatchingFiles(braille_path: str, mathml_path: str) -> tuple[list[str], list[str]]:
    """
    Reads lines from two files or directories and returns tuple of (braille_lines, mathml_lines).
    If directories are provided, matches files by base name (without extension) and combines all pairs.
    """
    braille_lines = []
    mathml_lines = []

    # Check if paths are directories
    if os.path.isdir(braille_path) and os.path.isdir(mathml_path):
        # Iterate through braille files and read matching pairs directly
        for filename in os.listdir(braille_path):
            if filename.endswith('.brls'):
                base_name = os.path.splitext(filename)[0]
                braille_file = os.path.join(braille_path, filename)
                mathml_file = os.path.join(mathml_path, base_name + '.mmls')

                # If matching mathml file exists, read both files
                if os.path.exists(mathml_file):
                    with open(braille_file, "r", encoding="utf-8") as f:
                        braille_lines.extend(f.read().splitlines())
                    with open(mathml_file, "r", encoding="utf-8") as f:
                        mathml_lines.extend(f.read().splitlines())
    else:
        # Handle as files (original behavior)
        with open(braille_path, "r", encoding="utf-8") as f:
            braille_lines = f.read().splitlines()
        with open(mathml_path, "r", encoding="utf-8") as f:
            mathml_lines = f.read().splitlines()

    return braille_lines, mathml_lines


def generate_examples(
    model: str,
    mathml_path: str,
    braille_path: str,
) -> list[dict[str, Any]]:
    """
    Reads MathML and Braille files to create a few-shot history.

    Args:
        model: The model identifier string (e.g., 'gemini-2.5-pro' or 'gpt-5-mini').
        mathml_path: Path to file with MathML strings (one per line).
        braille_path: Path to file with Braille strings (one per line).
    """
    history: list[dict[str, Any] | Any] = []  # Can contain dict for GPT or types.Content for Gemini
    is_gemini: bool = "gemini" in model.lower()
    try:
        with open(mathml_path, 'r', encoding='utf-8') as mathml_file, \
             open(braille_path, 'r', encoding='utf-8') as braille_file:

            # Use zip to pair mathml lines with their corresponding braille lines
            for math_line, braille_line in zip(mathml_file, braille_file):
                math_content = math_line.strip()
                braille_content = braille_line.strip()

                if not math_content or not braille_content:
                    continue

                # Using '1)' for all shots reinforces the Input=Output numbering pattern
                shot_num = "1)"

                if is_gemini:
                    # Format for Google Generative AI SDK (Gemini)
                    history.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"{shot_num} {math_line.strip()}")]
                    ))
                    history.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=f"{shot_num} {braille_line.strip()}")]
                    ))
                else:
                    # Format for OpenAI SDK (GPT)
                    history.append({
                        "role": "user",
                        "content": f"{shot_num} {math_content}"
                    })
                    history.append({
                        "role": "assistant",
                        "content": f"{shot_num} {braille_content}"
                    })

        return history
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e


def prepare_conversion_config(
    gen_braille: bool,
    braille_code: str,
    n_examples: int | None,
    start_index: int,
    n_tests: int | None,
    batch_size: int,
    ai_provider: str,
    service_tier: str,
    model: str,
    apiKeyName: str,
    symbol_mappings: list[SymbolMapping],
) -> tuple[RunConfig, list[dict[str, Any]], list[str], list[str]]:
    """
    Prepare configuration and data for a conversion run.

    Returns:
        Tuple of (config, instructions, examples, test_input, expected_output, model, apiKeyName)
    """
    ai_provider = ai_provider.lower()

    # File paths for examples
    example_braille_file = f"RustTestData/{braille_code}.brls"
    example_mathml_file = f"RustTestData/{braille_code}.mmls"

    if n_examples == 0:
        examples = []
    else:
        examples = generate_examples(
            model,
            example_braille_file,
            example_mathml_file
        )
        n_initial_examples = len(examples) // 2   # each example is a pair of braille and mathml
        if n_examples is None:
            n_examples = n_initial_examples
        if n_examples <= n_initial_examples:
            examples = examples[:2*n_examples]
        else:
            additional_examples = generate_examples(
                model,
                f"example_data/{braille_code.lower()}.brls",
                "example_data/mathml.mmls"
            )
            additional_examples = additional_examples[: 2*(n_examples - n_initial_examples)]
            examples.extend(additional_examples)
            n_examples = len(examples) // 2   # each example is a pair of braille and mathml

    # print(f"Examples len = {len(str(examples))}")
    # print(examples)
    # test_mathml_dir = "test_data/MathML"
    # test_braille_dir = f"test_data/{braille_code}"
    test_mathml_dir = "test_data/mathml.mmls"
    test_braille_dir = f"test_data/{braille_code}.brls"

    # File paths for input - gather lines from directories
    braille, mathml = readMatchingFiles(test_braille_dir, test_mathml_dir)
    if len(braille) != len(mathml):
        print("Error: Number of test inputs does not match number of expected outputs.")
        sys.exit(1)

    # Use n_tests parameter, default to len(mathml) if not provided
    print(f"n_tests: {n_tests}, len(mathml): {len(mathml)}")
    # Apply --start offset
    if start_index < 0:
        start_index = 0
    if start_index >= len(mathml):
        print(f"Error: start index {start_index} is beyond available tests ({len(mathml)}).")
        sys.exit(1)

    braille = braille[start_index:]
    mathml = mathml[start_index:]

    # Apply -t NUM limit
    n_tests_actual = min(n_tests, len(mathml)) if n_tests is not None else len(mathml)
    braille = braille[:n_tests_actual]
    mathml = mathml[:n_tests_actual]

    # GENERATE either braille or MathML
    test_input, expected = (mathml, braille) if gen_braille else (braille, mathml)

    # Create config
    config = RunConfig(
        braille_code=BrailleCode.NEMETH if braille_code.lower() == "nemeth" else BrailleCode.UEB,
        gen_braille=gen_braille,
        ai_provider=ai_provider,
        model=model,
        service_tier=service_tier,
        apiKeyName=apiKeyName,
        batch_size=batch_size,
        start_index=start_index,
        n_examples=n_examples,
        symbol_mappings=symbol_mappings,
        example_braille_file=example_braille_file,
        example_mathml_file=example_mathml_file,
        input_braille_file=test_braille_dir,
        input_mathml_file=test_mathml_dir
    )

    return config, examples, test_input, expected


def run_conversion(
    config: RunConfig,
    examples: list[dict[str, Any]],
    test_input: list[str],
    expected: list[str],
) -> None:
    """
    Run the conversion process to generate braille or MathML.

    Args:
        config: Pre-configured RunConfig object
        instructions: Instructions string for the model (without examples)
        examples: Examples (shots) for the model
        test_input: List of input strings to process (unnumbered)
        expected: List of expected output strings
        model: Model name to use
        apiKeyName: API key environment variable name
        batch_size: Batch size for processing
        ai_provider: AI provider name ("gemini" or "openai")
    """
    print(f"Using API key: {config.apiKeyName}")
    print(f"Generating {'braille' if config.gen_braille else 'MathML'} with {config.n_examples} examples, "
          f"{len(test_input)} tests with {config.model} for {config.braille_code}.")

    try:
        computed, total_tokens, total_generation_time = convert_input_with_model(
            config, examples, test_input
        )
        if computed is None:
            computed = []
        total_tokens['time'] = round(1000 * total_generation_time)  # ms -- needs to be an int
        output_filename = (
            f"{'to-' if config.gen_braille else 'from-'}{config.braille_code}-{config.model}-"
            f"{config.n_examples}exs-{config.start_index}-{config.start_index + len(test_input)}tests.txt"
        )
        write_results_to_file(test_input, computed, expected, total_tokens,
                              output_filename, config=config)
    except Exception as e:
        print(f"Conversion error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate braille or MathML using AI API (Gemini or "OpenAI")',
        epilog='''
Examples:
  # Generate MathML from Nemeth braille using Gemini, use 100 examples and 200 tests:
  python use_ai.py -ai gemini -e 100 -t 200 --config from-nemeth

  # Generate braille from MathML using "OpenAI", use all examples and all tests:
  python use_ai.py -ai gpt -e 9999 -t -1 --config to-ueb

  # Generate MathML from Nemeth braille using Gemini, use only rust examples and 50 tests:
  python use_ai.py -ai gemini -e -1 -t 50 -b 40 --config from-nemeth

Note:
  For Gemini: Requires GEMINI_API_KEY or GEMINI_PAID_API_KEY environment variable.
  For "OpenAI": Requires OPENAI_API_KEY environment variable.
  Set OPENAI_MODEL environment variable to override "OpenAI" default model (default: gpt-4o).
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-ai', '--ai-provider', type=str, required=True,
                        choices=['gemini', 'openai'],
                        help='AI provider: "gemini" or "openai" (case-insensitive)')
    parser.add_argument('-e', '--examples', type=int, required=True,
                        help=('Number of examples to use. A negative number means use all available examples.'))
    parser.add_argument(
        '-s', '--start',
        type=int,
        default=0,
        help='Start at test index NUM (0-based). Default: 0.'
    )
    parser.add_argument('-t', '--tests', type=int, required=True,
                        help=('Number of tests to process. A negative number means use all available tests.'))
    parser.add_argument('-b', '--batch-size', type=int, default=80,
                        help='Batch size for processing (default: 80).')
    parser.add_argument('--config', nargs='*', metavar='CONFIG',
                        help='Select configurations to run (case-insensitive). '
                             'Options: to-nemeth, to-ueb, from-nemeth, from-ueb. '
                             'If not specified, all configurations are run.')
    parser.add_argument(
        "--service-tier",
        choices=["auto", "flex"],
        default="flex",   # saves money, but slower than auto
        help="OpenAI service tier (auto or flex). Default: flex."
    )

    args = parser.parse_args()

    # Normalize AI provider
    ai_provider = args.ai_provider.lower()
    if ai_provider not in ['gemini', 'openai']:
        print(f"Error: Invalid AI provider '{args.ai_provider}'. "
              f"Must be 'gemini' or 'openai'.")
        sys.exit(1)

    # Set model and API key based on provider
    if ai_provider == "gemini":
        # model = "gemini-2.5-flash"   # for quick testings
        model = "gemini-2.5-pro"
        model = "gemma-4-31b-it"
        # model = "gemini-3.1-pro-preview"
        apiKeyName = "GEMINI_API_KEY"
        # apiKeyName = "GEMINI_PAID_API_KEY"
    elif ai_provider == "openai":
        model = "gpt-5-mini"
        # model = "gpt-5-nano"  # nano doesn't seem to understand braille instructions
        model = "gpt-5.4"
        apiKeyName = "OPENAI_API_KEY"
    else:
        raise ValueError(f"Unknown AI provider: {ai_provider}")

    # Convert negative numbers to None (meaning "all")
    n_examples = None if args.examples < 0 else args.examples
    n_tests = None if args.tests < 0 else args.tests

    # Map configuration strings to conversion parameters (case-insensitive)
    config_map = {
        'to-nemeth': (True, 'Nemeth'),
        'to-ueb': (True, 'UEB'),
        'from-nemeth': (False, 'Nemeth'),
        'from-ueb': (False, 'UEB'),
    }

    # All possible configurations
    all_conversion_params = [
        (True, 'Nemeth'),
        (True, 'UEB'),
        (False, 'Nemeth'),
        (False, 'UEB'),
    ]

    # Filter configurations based on provided arguments (case-insensitive)
    selected_configs = []
    if args.config:
        provided_configs = [c.lower() for c in args.config]
        valid_configs = {k.lower(): v for k, v in config_map.items()}

        invalid_configs = []
        for config_str in provided_configs:
            if config_str in valid_configs:
                if valid_configs[config_str] not in selected_configs:
                    selected_configs.append(valid_configs[config_str])
            else:
                invalid_configs.append(config_str)

        if invalid_configs:
            print(f"Error: Invalid configuration(s): {', '.join(invalid_configs)}")
            print(f"Valid options are: {', '.join(config_map.keys())}")
            sys.exit(1)

        if not selected_configs:
            print("Error: No valid configurations selected.")
            sys.exit(1)
    else:
        selected_configs = all_conversion_params

    symbol_mappings = load_symbol_mappings("Nemeth_charmap.yaml", "UEB_charmap.yaml")
    conversion_params = selected_configs

    # Prepare all configurations before asking for confirmation
    print("\n=== Preparing Configurations ===")
    configs_data = []

    for gen_braille, braille_code in conversion_params:
        try:
            config_data = prepare_conversion_config(
                    gen_braille, braille_code, n_examples, args.start, n_tests,
                    args.batch_size, ai_provider, args.service_tier, model,
                    apiKeyName, symbol_mappings
                )
            configs_data.append(config_data)
        except Exception as e:
            print(f"Error preparing config for {braille_code} ({'braille' if gen_braille else 'MathML'}): {e}")
            sys.exit(1)

    # Create threads for each conversion
    # Display first configuration
    print("\n=== Full Configuration ===")
    config, examples, test_input, expected = configs_data[0]
    conversion_type = f"{'Generate Braille' if config.gen_braille else 'Generate MathML'} ({config.braille_code})"
    print(f"\n--Configuration 1/{len(configs_data)}: {conversion_type} ---")
    config_str = config.print_config(n_tests=len(test_input), short=True)
    print(config_str)

    # Ask for confirmation once
    print("\n=== Confirmation ===")
    print("Is this correct? (y/yes to proceed, anything else to exit): ", end='', flush=True)
    response = input().strip().lower()
    confirmed = response in ('y', 'yes')
    if not confirmed:
        print("Exiting without processing.")
        sys.exit(0)
    threads = []
    for config, examples, test_input, expected in configs_data:
        thread = threading.Thread(
            target=run_conversion,
            args=(config, examples, test_input, expected)
        )
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
