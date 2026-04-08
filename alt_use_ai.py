from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any
import math
import json
import re
import time
import hashlib
import logging
import asyncio
import xml.etree.ElementTree as ET
import yaml
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from compare_mathml_in_csv import setMathCATPreferences, areCanonicallyEqual, CanonicalResults
from tqdm import tqdm
from ai_config import build_config_from_cli, ModelConfig
from openai import OpenAI, AzureOpenAI
import google.genai as genai
import argparse
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# ============================================================
# Core Data Structures
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


# ============================================================
# Triple-file Loaders (Examples + Tests)
# ============================================================

def load_tests_triple(
    mathml_path: str,
    nemeth_path: str,
    ueb_path: str
) -> tuple[list[str], list[str], list[str]]:
    """Generic triple-file loader for aligned MathML/Nemeth/UEB."""
    with open(mathml_path, "r", encoding="utf-8") as f_m:
        mathml = [line.strip() for line in f_m if line.strip()]

    with open(nemeth_path, "r", encoding="utf-8") as f_n:
        nemeth = [line.strip() for line in f_n if line.strip()]

    with open(ueb_path, "r", encoding="utf-8") as f_u:
        ueb = [line.strip() for line in f_u if line.strip()]

    if not (len(mathml) == len(nemeth) == len(ueb)):
        print(mathml_path)
        raise ValueError(
            f"Line count mismatch: MathML={len(mathml)}, "
            f"Nemeth={len(nemeth)}, UEB={len(ueb)}"
        )

    return mathml, nemeth, ueb


def load_math_examples_triple(
    mathml_path: str,
    nemeth_path: str,
    ueb_path: str
) -> list[MathExample]:
    """Build example objects from triple-file loader."""
    mathml, nemeth, ueb = load_tests_triple(mathml_path, nemeth_path, ueb_path)

    examples: list[MathExample] = []
    for i in range(len(mathml)):
        examples.append(MathExample(
            mathml=mathml[i],
            braille=nemeth[i],
            code=BrailleCode.NEMETH
        ))
        examples.append(MathExample(
            mathml=mathml[i],
            braille=ueb[i],
            code=BrailleCode.UEB
        ))

    return examples


def math_examples_from_data(
    mathml: list[str],
    nemeth: list[str],
    ueb: list[str],
) -> list[MathExample]:
    """Build example objects from in-memory data by zipping mathml and braille lists."""
    if len(mathml) != len(nemeth) or len(mathml) != len(ueb):
        raise ValueError(f"Length mismatch: mathml has {len(mathml)} items,"
                         f"nemeth has {len(nemeth)} items,"
                         f"ueb has {len(ueb)} items")

    examples: list[MathExample] = []
    for mml_str, nemeth_str, ueb_str in zip(mathml, nemeth, ueb):
        examples.append(MathExample(
            mathml=mml_str,
            braille=nemeth_str,
            code=BrailleCode.NEMETH
        ))
        examples.append(MathExample(
            mathml=mml_str,
            braille=ueb_str,
            code=BrailleCode.UEB
        ))

    return examples


def math_examples_from_isolated_data(
    mathml_path: str,
    braille_path: str,
    code: BrailleCode,
) -> list[MathExample]:
    """Build example objects from in-memory data by zipping mathml and braille lists."""
    with open(mathml_path, "r", encoding="utf-8") as f_m:
        mathml = [line.strip() for line in f_m]

    with open(braille_path, "r", encoding="utf-8") as f_n:
        braille = [line.strip() for line in f_n]

    if len(mathml) != len(braille):
        raise ValueError(f"Length mismatch: {mathml_path} has {len(mathml)} items !="
                         f"{braille_path} has {len(braille)} items != {len(mathml)}"
                         )

    examples: list[MathExample] = []
    for i in range(len(mathml)):
        if mathml[i] != "" and braille[i] != "":
            examples.append(MathExample(
                mathml=mathml[i],
                braille=braille[i],
                code=code
            ))

    return examples


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


# ============================================================
# Context Rules (MathML side only)
# ============================================================

def build_context_rules_block(used: set[str], code: BrailleCode) -> str:
    lines = []
    if code is BrailleCode.NEMETH:
        if "," in used:
            lines.append("- ',' inside a number → ⠐⠂.")
        if "." in used:
            lines.append("- '.' inside a number → ⠨.")
    return "\n".join(lines) + ("\n" if lines else "")


def build_structural_context_block(notes: list[str]) -> str:
    if not notes:
        return ""
    return "Structural context:\n" + "\n".join(f"- {n}" for n in notes) + "\n\n"


# ============================================================
# Embedding Caching
# ============================================================

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def hash_examples(examples: list[MathExample]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(ex.mathml.encode("utf-8"))
        h.update(ex.braille.encode("utf-8"))
        h.update(ex.code.value.encode("utf-8"))
    return h.hexdigest()


def save_embeddings(path: str, embeddings: list[list[float]], hash_value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"hash": hash_value, "embeddings": embeddings}, f)


def load_embeddings(path: str) -> tuple[str, list[list[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["hash"], data["embeddings"]


def compute_embeddings(
    ai_config: ModelConfig,
    items: list[str],
    chunk_size: int = 1000
) -> list[list[float]]:
    """
    Compute embeddings for a large list of strings in safe chunks.
    Returns embeddings in the same order as `items`.
    """

    if ai_config.provider == "gemini":
        raise RuntimeError("Gemini does not support embeddings.")

    all_embeddings: list[list[float]] = []

    for start in range(0, len(items), chunk_size):
        end = start + chunk_size
        chunk = items[start:end]

        resp = ai_config.client.embeddings.create(
            model=ai_config.embedding_model,
            input=chunk
        )

        # Append embeddings in order
        for item in resp.data:
            all_embeddings.append(item.embedding)

    return all_embeddings


def embedding_cache_filename(
    ai_config: ModelConfig,
    cache_dir_path: str,
    cache_path: str,
    use_mathml: bool
) -> str:
    """
    Build a deterministic cache filename inside `cache_dir_path`.

    Components:
      - provider (openai / azure / gemini)
      - embedding model (normalized)
      - direction (mathml / braille)
      - original cache_path (normalized)
    """
    direction = "mathml" if use_mathml else "braille"

    # Normalize the embedding model for filesystem safety
    model_safe = ai_config.embedding_model.replace("/", "_") or "no-embeddings"

    # Normalize the user-provided cache_path (strip dirs, replace slashes)
    base = os.path.basename(cache_path).replace("/", "_")

    # Build deterministic filename
    filename = f"embeddings_{ai_config.provider}_{model_safe}_{direction}_{base}"

    # Ensure directory exists
    os.makedirs(cache_dir_path, exist_ok=True)

    return os.path.join(cache_dir_path, filename)


def get_or_compute_embeddings(
    ai_config: ModelConfig,
    examples: list[MathExample],
    cache_dir_path: str,
    use_mathml: bool
) -> list[list[float]]:

    # Gemini: no embeddings → return empty list
    if ai_config.provider == "gemini":
        return []

    # Build deterministic provider/model/direction-aware cache file
    cache_file = embedding_cache_filename(
        ai_config=ai_config,
        cache_dir_path=cache_dir_path,
        cache_path="embeddings.json",   # original name preserved in filename
        use_mathml=use_mathml
    )

    current_hash = hash_examples(examples)

    try:
        stored_hash, stored_embeddings = load_embeddings(cache_file)
        if stored_hash == current_hash:
            logging.info(f"Loaded cached embeddings: {cache_file}")
            return stored_embeddings
        else:
            logging.info(f"Embedding cache invalid: {cache_file}")
    except FileNotFoundError:
        logging.info(f"No embedding cache found: {cache_file}")

    # Compute fresh embeddings
    texts = [ex.mathml if use_mathml else ex.braille for ex in examples]
    embeddings = compute_embeddings(ai_config, items=texts)

    save_embeddings(cache_file, embeddings, current_hash)
    return embeddings


# ============================================================
# Retrieval (with symbol overlap for both directions)
# ============================================================

def extract_symbols_from_example_mathml(example: MathExample) -> set[str]:
    try:
        return extract_symbols_from_mathml(example.mathml)
    except Exception:
        return set()


def compute_symbol_overlap_score(query: set[str], example: set[str]) -> float:
    if not query or not example:
        return 0.0
    overlap = len(query & example)
    return overlap / len(query)


def extract_braille_cells(braille: str) -> set[str]:
    return set(braille)


def compute_braille_symbol_overlap(
    query_cells: set[str],
    example_braille: str
) -> float:
    ex_cells = extract_braille_cells(example_braille)
    if not query_cells or not ex_cells:
        return 0.0
    overlap = len(query_cells & ex_cells)
    return overlap / len(query_cells)


def retrieve_top_k_indices(
    query_embedding: list[float],
    query_symbols: set[str],
    all_embeddings: list[list[float]],
    examples: list[MathExample],
    code: BrailleCode,
    k: int,
    use_symbol_overlap: Literal["mathml", "braille", "none"]
) -> list[int]:
    scored = []
    for i, emb in enumerate(all_embeddings):
        if examples[i].code != code:
            continue
        cos = cosine_similarity(query_embedding, emb)
        if use_symbol_overlap == "mathml":
            ex_symbols = extract_symbols_from_example_mathml(examples[i])
            sym = compute_symbol_overlap_score(query_symbols, ex_symbols)
        elif use_symbol_overlap == "braille":
            sym = compute_braille_symbol_overlap(query_symbols, examples[i].braille)
        else:
            sym = 0.0
        score = 0.9 * cos + 0.1 * sym
        scored.append((score, i))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [idx for _, idx in scored[:k]]


# ============================================================
# Adaptive k-selection
# ============================================================

def estimate_expression_complexity(used: set[str], notes: list[str]) -> str:
    has_greek = any("\u0370" <= ch <= "\u03FF" for ch in used)
    has_integral = any(ch in {"∫", "∑", "∏"} for ch in used)
    has_matrix = any(ch in {"[", "]", "|"} for ch in used)
    if has_matrix or has_integral:
        return "complex"
    if has_greek or len(used) > 10 or notes:
        return "medium"
    return "simple"


def choose_k_values(used: set[str], query_length: int) -> tuple[int, int]:
    complexity = len(used) + query_length   # number of distinct symbols along with a lenght component
    if complexity < 5:
        return 10, 20
    if complexity < 10:
        return 20, 40
    return 40, 80


# ============================================================
# Prompt Builders (Unified)
# ============================================================

def build_prompt(
    query_input: str,
    examples: list[MathExample],
    example_indices: list[int],
    code: BrailleCode,
    symbol_block: str,
    context_rules_block: str,
    structural_block: str,
    generate_braille: bool
) -> str:

    if generate_braille:
        header = (
            "You are an expert MathML to {code.value.upper()} Braille translator.\n"
            f"Use the pairs of examples to infer the correct mapping from MathML to {code.value.upper()} Braille.\n\n"
        )
        if len(examples) == 0:
            examples_text = ""
        else:
            examples_text = "".join(
                f"MathML:\n{examples[i].mathml}\n"
                f"Braille ({code.value}):\n{examples[i].braille}\n\n"
                for i in example_indices
            )
        query_block = (
            f"Now translate the following MathML into {code.value.upper()} Braille.\n"
            f"MathML:\n{query_input}\n\n"
            "Return ONLY Unicode braille characters. "
            "It is important to pay attention to generating Unicode braille spaces when needed in the braille. "
            "Relational operators such as <, >, ≤, ≥, =, ≠ almost always need to have spaces on the left and right "
            "unless they are in a script position.\n"

        )
        if code == BrailleCode.NEMETH:
            query_block += ("Some things to remember about Nemeth Braille: \n"
                            "- the number sign indicator ⠼ that precedes digits is ONLY needed in these two cases: \n"
                            "  1. at the start of a line, after a space, or after punctuation. \n"
                            "  2. if a digit follows a minus sign that is at "
                            " the start of a line, after a space, or after punctuation.\n"
                            "- the English letter indicator ⠰ that precedes Roman letters is ONLY needed "
                            "in these two cases: \n"
                            "  1. at the start of a line, after punctuation, or after whitespace.\n"
                            "  2. if a letter follows a minus sign that is at the start of a line, after a space, "
                            "or after punctuation on the left and right of the letter, ignoring any intervening open "
                            "or close characters.\n"
                            "- the English letter indicator ⠰ is never needed between two Roman letters.\n"
                            "- a letter with an integer subscript should NOT use a subscript indicator "
                            "if the subscript is not inside of a subscript or superscript.")
        else:  # UEB
            query_block += (
                "Some things to remember about UEB Braille:\n"
                " - grade 1 indicators ⠰, grade 1 word indicators ⠰⠰, and grade 1 passage indicators ⠰⠰⠰ are "
                "often needed at the start of a line.\n"
                " - in general, you want to minimize the use of grade 1 indicators (each ⠰ counts as an instance of "
                "a grade 1 indicator). Choose word or passage indicators when they result in fewer grade 1 "
                "indicators. If there are more than two braille spaces (⠀), use the grade 1 passage indicators.\n"
                "- A letter or unbroken sequence of letters is 'standing alone' if the symbols before and after the "
                "letter or sequence are spaces, hyphens, dashes, or any combination thereof, including some common "
                "punctuation. An opening bracketing character before a sequence or closing bracketing character after "
                "a sequence should be included in the above definition of 'standing alone'. A single letter (excluding "
                "a, i and o) is considered 'standing alone' if it is preceded by a space.\n"
                "- A grade 1 indicator ⠰ is needed before a standing alone letter or sequence of letters.\n"
                "- the number sign indicator ⠼ is ONLY needed before digits and starts numeric mode.\n"
                "- All fraction, root, subscript, superscript, etc., indicators MUST use grade 1 mode.\n"
                "- Numeric mode includes the digits and the fraction line (⠌) for simple numeric fractions. It also "
                "includes ',', '.', and spaces when they appear inside of MathML mn elements.\n"
                "- if the lowercase letters a-j follow a digit, you MUST use a grade 1 indicator ⠰ before the letter.\n"
                "- numeric fraction do not use start or end fraction indicators, but all other fractions start with ⠷, "
                "end with ⠾, and use ⠨⠌ as the fraction bar.\n"
                "- all subscripts MUST start with the subscript indicator ⠢.\n"
            )
    else:
        header = (
            f"You are a expert {code.value.upper()} Braille to MathML translator.\n"
            f"Use the pairs of examples to infer the correct mapping from {code.value.upper()} Braille to MathML.\n\n"
        )
        if len(examples) == 0:
            examples_text = ""
        else:
            examples_text = "".join(
                f"Braille ({code.value}):\n{examples[i].braille}\n"
                f"MathML:\n{examples[i].mathml}\n\n"
                for i in example_indices
            )
        query_block = (
            f"Now translate the following {code.value.upper()} Braille into MathML.\n"
            f"Braille:\n{query_input}\n\n"
            "Return ONLY valid MathML markup."
        )

    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(f"examples_text: {examples_text}\n")
        f.write(f"query_block: {query_block}\n\n\n\n")

    return (
        header
        + examples_text
        + symbol_block
        + context_rules_block
        + structural_block
        + query_block
    )


# ============================================================
# GPT Call
# ============================================================
# One shared pool for all GPT calls
_GPT_POOL = ThreadPoolExecutor(max_workers=8)


def _do_gpt_call(
    ai_config: ModelConfig,
    prompt: str,
    temperature: float,
) -> tuple[str, int, int, int]:
    """
    Execute a single GPT call for OpenAI, Azure OpenAI, or Gemini.

    Returns:
        (text, input_tokens, output_tokens, total_tokens)
    """

    # ------------------------------------------------------------
    # OpenAI + Azure OpenAI (same API shape)
    # ------------------------------------------------------------
    if isinstance(ai_config.client, OpenAI) or isinstance(ai_config.client, AzureOpenAI):
        resp = ai_config.client.responses.create(
            model=ai_config.model,
            input=prompt,
            temperature=temperature,
            service_tier=ai_config.service_tier,
        )
        text = resp.output_text or ""   # narrow str | None → str
        usage = resp.usage
        return (
            text,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )

    # Gemini (google.genai)
    if isinstance(ai_config.client, genai.Client):
        resp = ai_config.client.models.generate_content(
            model=ai_config.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        text = resp.text or ""   # narrow str | None → str
        return text, 0, 0, 0

    # Unknown provider
    raise ValueError(f"Unsupported client type: {type(ai_config.client)}")


def call_gpt_with_retry(
    ai_config: ModelConfig,
    messages: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> tuple[str, int, int, int]:
    timeout = 200.0 if ai_config.service_tier == "auto" else 500.0
    for attempt in range(max_retries):
        future = _GPT_POOL.submit(_do_gpt_call, ai_config, messages, temperature)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            print(f"GPT call timed out (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"GPT call failed (attempt {attempt+1}/{max_retries}): {e}")

        if attempt == max_retries - 1:
            raise RuntimeError("All retry attempts failed")
        time.sleep(retry_delay)

    # This should never be reached, but satisfies type checker
    raise RuntimeError("Unexpected exit from retry loop")


# ============================================================
# Unified Translation Core
# ============================================================

def translate(
    ai_config: ModelConfig,
    query_input: str,
    query_embedding: list[float],
    examples: list[MathExample],
    mathml_embeddings: list[list[float]],
    braille_embeddings: list[list[float]],
    code: BrailleCode,
    symbol_mappings: list[SymbolMapping],
    generate_braille: bool,
    k_primary: int,
    k_fallback: int,
    similarity_threshold: float,
    primary_temperature: float = 0.0,
    fallback_temperature: float = 0.0,
    final_temperature: float = 0.0,
) -> tuple[str, int, int, int]:
    """
    Full translation pipeline:
        - retrieval
        - example selection
        - message construction
        - GPT call
        - returns (text, input_tokens, output_tokens, total_tokens)
    """
    if generate_braille:
        used_symbols = extract_symbols_from_mathml(query_input)
        query_length = 2 * count_complicated_elements(query_input)    # these typically add two braille cells or more
        structural_notes = extract_structural_context(query_input)
        all_embeddings = mathml_embeddings
        overlap_mode: Literal["mathml", "braille", "none"] = "mathml"
    else:
        used_symbols: set[str] = set(query_input)
        query_length = len(query_input)  # braille is only 64 symbols, so doesn't really reflect complexity
        structural_notes = []
        all_embeddings = braille_embeddings
        overlap_mode = "braille"

    symbol_block = build_symbol_block(used_symbols, symbol_mappings, code, generate_braille)
    context_rules_block = build_context_rules_block(used_symbols, code) if generate_braille else ""
    structural_block = build_structural_context_block(structural_notes) if generate_braille else ""

    k_primary, k_fallback = choose_k_values(used_symbols, query_length)
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(f"In '{query_input}\n== used_symbols: {len(used_symbols)}, query_length: {query_length},"
                f" examples to use: k_primary/fallback: {k_primary}/{k_fallback}\n"
                f"char mappings:\n{symbol_block}\n")
    # k_primary = min(k_primary, base_k_primary)
    # k_fallback = min(k_fallback, base_k_fallback)

    top_primary = retrieve_top_k_indices(
        query_embedding=query_embedding,
        query_symbols=used_symbols,
        all_embeddings=all_embeddings,
        examples=examples,
        code=code,
        k=k_primary,
        use_symbol_overlap=overlap_mode
    )

    best_primary_sim = (
        cosine_similarity(query_embedding, all_embeddings[top_primary[0]])
        if top_primary else 0.0
    )

    if best_primary_sim >= similarity_threshold and top_primary:
        prompt = build_prompt(
            query_input,
            examples,
            top_primary,
            code,
            symbol_block,
            context_rules_block,
            structural_block,
            generate_braille
        )
        return call_gpt_with_retry(ai_config, prompt, temperature=primary_temperature)

    logging.info(
        "Primary fallback triggered (code=%s, direction=%s, sim=%.3f)",
        code.value,
        "MathML→Braille" if generate_braille else "Braille→MathML",
        best_primary_sim
    )

    top_fallback = retrieve_top_k_indices(
        query_embedding=query_embedding,
        query_symbols=used_symbols,
        all_embeddings=all_embeddings,
        examples=examples,
        code=code,
        k=k_fallback,
        use_symbol_overlap=overlap_mode
    )

    if top_fallback:
        prompt_fb = build_prompt(
            query_input,
            examples,
            top_fallback,
            code,
            symbol_block,
            context_rules_block,
            structural_block,
            generate_braille
        )
        fb_output = call_gpt_with_retry(ai_config, prompt_fb, temperature=fallback_temperature)
        if fb_output:
            logging.info(
                "Fallback retrieval succeeded (code=%s, direction=%s)",
                code.value,
                "MathML→Braille" if generate_braille else "Braille→MathML"
            )
            return fb_output

    logging.info(
        "Final rule-based fallback used (code=%s, direction=%s)",
        code.value,
        "MathML→Braille" if generate_braille else "Braille→MathML"
    )

    final_prompt = build_prompt(
        query_input,
        [],
        [],
        code,
        symbol_block,
        context_rules_block,
        structural_block,
        generate_braille
    )
    return call_gpt_with_retry(ai_config, final_prompt, temperature=final_temperature)


# ============================================================
# Async wrappers + unified parallel batch
# ============================================================
async def translate_single_async(
    ai_config: ModelConfig,
    query_input: str,
    query_embedding: list[float],
    examples: list[MathExample],
    mathml_embeddings: list[list[float]],
    braille_embeddings: list[list[float]],
    code: BrailleCode,
    symbol_mappings: list[SymbolMapping],
    generate_braille: bool,
    k_primary: int,
    k_fallback: int,
    similarity_threshold: float,
    primary_temperature: float = 0.0,
    fallback_temperature: float = 0.1,
    final_temperature: float = 0.0,
    timeout: float = 30.0,
) -> tuple[str, int, int, int]:

    loop = asyncio.get_running_loop()

    # Run translate() in executor, but apply timeout at the async layer
    return await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: translate(
                ai_config,
                query_input,
                query_embedding,
                examples,
                mathml_embeddings,
                braille_embeddings,
                code,
                symbol_mappings,
                generate_braille,
                k_primary,
                k_fallback,
                similarity_threshold,
                primary_temperature,
                fallback_temperature,
                final_temperature,
            ),
        ),
        timeout=timeout,
    )


async def parallel_batch_translate(
    ai_config: ModelConfig,
    query_inputs: list[str],
    query_embeddings: list[list[float]],
    examples: list[MathExample],
    mathml_embeddings: list[list[float]],
    braille_embeddings: list[list[float]],
    code: BrailleCode,
    symbol_mappings: list[SymbolMapping],
    generate_braille: bool,
    k_primary: int,
    k_fallback: int,
    similarity_threshold: float,
    num_workers: int = 8
) -> tuple[list[str], dict[str, int]]:
    """
    Runs many translations in parallel using async workers.

    Returns:
        (results, totals)
        where totals = {"input": int, "output": int, "total": int}
    """

    # -----------------------------
    # Setup shared structures
    # -----------------------------
    queue: asyncio.Queue[
        tuple[int, str, list[float]] | None
    ] = asyncio.Queue()

    results: list[str | None] = [None] * len(query_inputs)
    lock = asyncio.Lock()

    totals: dict[str, int] = {"input": 0, "output": 0, "total": 0}

    # -----------------------------
    # Populate queue
    # -----------------------------
    if not query_embeddings:
        # Gemini or embedding-free mode
        for i, q in enumerate(query_inputs):
            queue.put_nowait((i, q, []))
    else:
        for i, (q, emb) in enumerate(zip(query_inputs, query_embeddings)):
            queue.put_nowait((i, q, emb))

    # Add sentinel None for each worker
    for _ in range(num_workers):
        queue.put_nowait(None)

    # -----------------------------
    # Worker definition
    # -----------------------------
    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return

            index, q_input, q_emb = item

            # Run translation in thread pool
            text, in_tok, out_tok, tot_tok = await translate_single_async(
                ai_config,
                q_input,
                q_emb,
                examples,
                mathml_embeddings,
                braille_embeddings,
                code,
                symbol_mappings,
                generate_braille,
                k_primary,
                k_fallback,
                similarity_threshold,
                primary_temperature=0.0,
                fallback_temperature=0.1,
                final_temperature=0.0,
            )

            # Store result + accumulate totals
            async with lock:
                results[index] = text
                totals["input"] += in_tok
                totals["output"] += out_tok
                totals["total"] += tot_tok

            queue.task_done()

    # -----------------------------
    # Launch workers
    # -----------------------------
    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

    # -----------------------------
    # Progress bar loop
    # -----------------------------
    with tqdm(total=len(query_inputs)) as pbar:
        prev_done = 0
        while any(not w.done() for w in workers):
            done = sum(r is not None for r in results)
            pbar.update(done - prev_done)
            prev_done = done
            await asyncio.sleep(0.1)

    # Ensure queue is empty
    await queue.join()

    # -----------------------------
    # Return final results
    # -----------------------------
    return [r for r in results if r is not None], totals


# ============================================================
# Evaluation + Round-trip
# ============================================================

def evaluate(name: str, results: list[str], expected: list[str]) -> None:
    total = len(expected)
    correct = sum(1 for a, b in zip(results, expected) if a == b)
    print(f"{name}: {correct}/{total} correct ({correct/total:.2%})")


def round_trip_consistency(
    forward_outputs: list[str],
    reverse_outputs: list[str],
    original_inputs: list[str]
) -> float:
    total = len(original_inputs)
    correct = sum(
        1 for orig, rev in zip(original_inputs, reverse_outputs)
        if orig == rev
    )
    return correct / total if total else 0.0


def write_results_to_file(mode: str,
                          inputs: list[str],
                          computed_output: list[str],
                          expected_output: list[str],
                          n_examples: int,
                          first_test_index: int,
                          last_test_index: int,
                          info: dict[str, int],  # time is in ms
                          output_file: str,
                          ai_config: ModelConfig) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    """
    usage_info = str(info)[1:-1].replace("'", "").replace(": ", "=")
    print(f"Generated {len(computed_output)} outputs. Stats: {usage_info}ms")
    print(f"AI provider: {ai_config.provider}, model: {ai_config.model}")
    is_mathml_output = expected_output[0].startswith('<math')
    if len(computed_output) == 0:
        print("!!!No computed outputs to write.")
        return
    if is_mathml_output and not computed_output[0].startswith('<math'):
        print("Computed output does not appear to be MathML--first 5 lines:\n", computed_output[:5])
        return
    if not is_mathml_output and not re.match('[\u2800-\u28ff]', computed_output[0][0]):
        print("Computed output does not appear to be MathML--last 5 lines:\n", computed_output[len(computed_output)-5:])
        return

    # initial MathCAT
    setMathCATPreferences({})

    with open(output_file, "w", encoding="utf-8") as f:
        # Write variable values from main() at the start
        f.write(f"# {mode}: "
                f"Using {n_examples} examples and tests {first_test_index}-{last_test_index} of {len(inputs)} tests.\n")

        match_count = 0
        f.write(f"# {len(computed_output)} items. Usage info: {usage_info}ms\n#\n")
        if is_mathml_output:
            f.write("\n# NOT Normalized MathML\n")
        f.write("# Match | Test Input | Expected | Computed\n")
        for tests, computed, expected in zip(inputs, computed_output, expected_output):
            try:
                if is_mathml_output:
                    checked = areCanonicallyEqual(expected, computed)
                else:
                    checked = CanonicalResults(expected.strip().strip(' ') == computed.strip().strip(' '), "", "")
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
            for tests, computed, expected in zip(inputs, computed_output, expected_output):
                try:
                    checked = areCanonicallyEqual(expected, computed)
                except Exception as e:
                    print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                    checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)

                match = "✓" if checked.isEqual else "✗"
                f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        f.write(f"# Matches: {match_count} out of {len(computed_output)}: "
                f"{(match_count/len(computed_output)*100):.0f}%.")
        print(f"Matches: {match_count} out of {len(computed_output)}: {(match_count/len(computed_output)*100):.0f}%. "
              f"Results written to {output_file}. ")


# ============================================================
# CLI handling
# ============================================================

VALID_MODES = {
    "to-nemeth",
    "to-ueb",
    "from-nemeth",
    "from-ueb",
    "all",
}


def parse_cli_args() -> tuple[
    set[str], int | None, int | None, str, Literal["auto", "default", "flex", "scale", "priority"]
]:
    parser = argparse.ArgumentParser(
        description="MathML ↔ Braille translation test runner"
    )

    parser.add_argument(
        "modes",
        nargs="+",
        help="One or more of: to-nemeth, to-ueb, from-nemeth, from-ueb, all",
    )

    parser.add_argument(
        "-t", "--tests",
        type=int,
        default=None,
        help="Number of tests to run"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Skip the first NUM tests before running"
    )

    parser.add_argument(
        "-ai", "--ai_provider",
        type=str,
        choices=["openai", "azure", "gemini"],
        default="openai",
        help="Which LLM provider to use",
    )

    parser.add_argument(
        "--service-tier",
        choices=["auto", "flex"],
        default="flex",   # saves money, but slower than auto
        help="OpenAI service tier (auto or flex). Default: flex."
    )

    args = parser.parse_args()

    modes = {m.lower() for m in args.modes}

    if "all" in modes:
        modes = {"to-nemeth", "to-ueb", "from-nemeth", "from-ueb"}

    invalid = modes - VALID_MODES
    if invalid:
        raise ValueError(f"Invalid mode(s): {invalid}")

    service_tier: Literal[
        "auto", "default", "flex", "scale", "priority"
    ] = args.service_tier.lower()  # type: ignore[assignment]

    return (
        modes,
        args.tests,
        args.start,
        args.ai_provider,
        service_tier,
    )


def run_tests(
    modes: set[str],
    n_tests: int | None,
    test_start: int | None,
    ai_config: ModelConfig,
) -> None:
    # ---------------------------------------------------------
    # Load examples and tests starting with the RustTestData
    # ---------------------------------------------------------
    examples = math_examples_from_isolated_data(
        "RustTestData/Nemeth-cnclz.mmls",
        "RustTestData/Nemeth.brls",
        BrailleCode.NEMETH
    )
    examples.extend(math_examples_from_isolated_data(
        mathml_path="RustTestData/UEB-cnclz.mmls",
        braille_path="RustTestData/UEB.brls",
        code=BrailleCode.UEB
    ))

    examples.extend(load_math_examples_triple(
        "example_data/canonical-mathml.mmls",
        "example_data/nemeth.brls",
        "example_data/ueb.brls",
    ))

    test_mathml, test_nemeth, test_ueb = load_tests_triple(
        "test_data/canonical-mathml.mmls",
        "test_data/nemeth.brls",
        "test_data/ueb.brls",
    )

    # Apply --start offset
    if test_start is not None:
        test_mathml = test_mathml[test_start:]
        test_nemeth = test_nemeth[test_start:]
        test_ueb = test_ueb[test_start:]
    else:
        test_start = 0

    # Apply -t NUM limit
    if n_tests is not None:
        test_mathml = test_mathml[:n_tests]
        test_nemeth = test_nemeth[:n_tests]
        test_ueb = test_ueb[:n_tests]
    else:
        n_tests = len(test_mathml) - test_start

    # ---------------------------------------------------------
    # Load symbol mappings and embeddings
    # ---------------------------------------------------------
    symbol_mappings = load_symbol_mappings("Nemeth_charmap.yaml", "UEB_charmap.yaml")

    mathml_embeddings = get_or_compute_embeddings(
        ai_config,
        examples,
        cache_dir_path="example_data",
        use_mathml=True
    )
    braille_embeddings = get_or_compute_embeddings(
        ai_config,
        examples,
        cache_dir_path="example_data",
        use_mathml=False
    )

    test_examples = math_examples_from_data(test_mathml, test_nemeth, test_ueb)
    n_examples = len(test_examples)

    if 'to-nemeth' in modes or 'to-ueb' in modes:
        test_mathml_embeddings = get_or_compute_embeddings(
            ai_config,
            test_examples,
            cache_dir_path="test_data",
            use_mathml=True
        )
    if 'from-nemeth' in modes or 'from-ueb' in modes:
        test_braille_embeddings = get_or_compute_embeddings(
            ai_config,
            test_examples,
            cache_dir_path="test_data",
            use_mathml=False
        )
    one_based_test_start = test_start + 1 if test_start == 0 else test_start
    output_path_suffix = f"gpt-5.4-{one_based_test_start}-{one_based_test_start+n_tests-1}-tests.txt"

    # ---------------------------------------------------------
    # Translation configurations
    # ---------------------------------------------------------
    translation_configs = [
        {
            "mode": "to-nemeth",
            "inputs": test_mathml,
            "expected_outputs": test_nemeth,
            "code": BrailleCode.NEMETH,
            "generate_braille": True,
            "output_path": f"to_nemeth-{output_path_suffix}",
            "evaluate_label": "MathML → Nemeth",
            "result_var": "to_nemeth_results",
        },
        {
            "mode": "to-ueb",
            "inputs": test_mathml,
            "expected_outputs": test_ueb,
            "code": BrailleCode.UEB,
            "generate_braille": True,
            "output_path": f"to_ueb_{output_path_suffix}",
            "evaluate_label": "MathML → UEB",
            "result_var": "to_ueb_results",
        },
        {
            "mode": "from-nemeth",
            "inputs": test_nemeth,
            "expected_outputs": test_mathml,
            "code": BrailleCode.NEMETH,
            "generate_braille": False,
            "output_path": f"from_nemeth_{output_path_suffix}",
            "evaluate_label": "Nemeth → MathML",
            "result_var": "from_nemeth_results",
        },
        {
            "mode": "from-ueb",
            "inputs": test_ueb,
            "expected_outputs": test_mathml,
            "code": BrailleCode.UEB,
            "generate_braille": False,
            "output_path": f"from_ueb_{output_path_suffix}",
            "evaluate_label": "UEB → MathML",
            "result_var": "from_ueb_results",
        },
    ]

    # ---------------------------------------------------------
    # Storage for results + totals
    # ---------------------------------------------------------
    results: dict[str, list[str] | None] = {}
    totals_by_mode: dict[str, dict[str, int]] = {}

    # ---------------------------------------------------------
    # Run each translation mode
    # ---------------------------------------------------------
    for config in translation_configs:
        mode = config["mode"]

        if mode in modes:
            outputs, totals = asyncio.run(
                parallel_batch_translate(
                    ai_config=ai_config,
                    query_inputs=config["inputs"],
                    query_embeddings=(
                        test_mathml_embeddings if mode in ['to-nemeth', 'to-ueb']
                        else test_braille_embeddings
                    ),
                    examples=examples,
                    mathml_embeddings=mathml_embeddings,
                    braille_embeddings=braille_embeddings,
                    code=config["code"],
                    symbol_mappings=symbol_mappings,
                    generate_braille=config["generate_braille"],
                    k_primary=5,
                    k_fallback=10,
                    similarity_threshold=0.75,
                    num_workers=5
                )
            )

            outputs = [line.replace("\n", "").replace(" ", "") for line in outputs]
            results[config["result_var"]] = outputs
            totals_by_mode[mode] = totals
            write_results_to_file(
                mode,
                config["inputs"],
                outputs,
                config["expected_outputs"],
                n_examples,
                test_start,
                test_start + n_tests,
                totals,
                config["output_path"],
                ai_config
            )

            # Evaluate accuracy
            evaluate(config["evaluate_label"], outputs, config["expected_outputs"])

        else:
            results[config["result_var"]] = None

    # ---------------------------------------------------------
    # Round-trip tests
    # ---------------------------------------------------------
    to_nemeth_results = results.get("to_nemeth_results")
    to_ueb_results = results.get("to_ueb_results")

    round_trip_configs = [
        {
            "forward_results": to_nemeth_results,
            "forward_mode": "from-nemeth",
            "code": BrailleCode.NEMETH,
            "output_path": "tmp_roundtrip_nemeth.jsonl",
            "label": "MathML ↔ Nemeth",
        },
        {
            "forward_results": to_ueb_results,
            "forward_mode": "from-ueb",
            "code": BrailleCode.UEB,
            "output_path": "tmp_roundtrip_ueb.jsonl",
            "label": "MathML ↔ UEB",
        },
    ]

    for rt_config in round_trip_configs:
        forward_results = rt_config["forward_results"]
        forward_mode = rt_config["forward_mode"]

        if forward_results is not None and forward_mode in modes:
            rt_reverse, rt_totals = asyncio.run(
                parallel_batch_translate(
                    ai_config=ai_config,
                    query_inputs=forward_results,
                    query_embeddings=(
                        test_mathml_embeddings if forward_mode in ['to-nemeth', 'to-ueb']
                        else test_braille_embeddings
                    ),
                    examples=examples,
                    mathml_embeddings=mathml_embeddings,
                    braille_embeddings=braille_embeddings,
                    code=rt_config["code"],
                    symbol_mappings=symbol_mappings,
                    generate_braille=False,
                    k_primary=5,
                    k_fallback=10,
                    similarity_threshold=0.75,
                    num_workers=8
                )
            )

            rt_score = round_trip_consistency(forward_results, rt_reverse, test_mathml)
            print(f"Round-trip consistency ({rt_config['label']}): {rt_score:.2%}")


def main():
    (
        modes,
        num_tests,
        start_index,
        provider,
        service_tier,
    ) = parse_cli_args()

    ai_config = build_config_from_cli(provider, service_tier)

    run_tests(
        modes,
        num_tests,
        start_index,
        ai_config,
    )


if __name__ == "__main__":
    main()
