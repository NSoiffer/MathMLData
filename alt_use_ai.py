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
from openai import OpenAI
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')   # Ensure UTF-8 output for Unicode Braille

client = OpenAI()

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
    for mml, nemeth, ueb in zip(mathml, nemeth, ueb):
        examples.append(MathExample(
            mathml=mml,
            braille=nemeth,
            code=BrailleCode.NEMETH
        ))
        examples.append(MathExample(
            mathml=mml,
            braille=ueb,
            code=BrailleCode.UEB
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
        if tag in {"mo", "mi", "mn"} and elem.text:
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
            if "," in elem.text:
                notes.append("Comma inside <mn> → numeric comma.")
            if "-" in elem.text:
                notes.append("Minus inside <mn> → unary or numeric.")
        if tag == "mo" and elem.text:
            if elem.text == ",":
                notes.append("Comma as <mo> → argument separator.")
            if elem.text == "-":
                notes.append("Minus as <mo> → binary subtraction.")
    return sorted(set(notes))


# ============================================================
# Symbol Mapping Blocks
# ============================================================
def _extract_first_t(obj: any) -> str | None:
    if isinstance(obj, dict):
        if "t" in obj:
            return obj["t"]

        if "test" in obj:
            tb: dict[str, any] = obj["test"]

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
        if any(ch.isdigit() for ch in used) or "," in used or "." in used:
            lines.append("Nemeth numeric mode rules:")
            lines.append("- Numeric mode begins with ⠼.")
            lines.append("- Inside numeric mode, digits use standard Nemeth digits.")
            if "," in used:
                lines.append("- ',' inside numeric mode → ⠐⠂.")
            if "." in used:
                lines.append("- '.' inside numeric mode → ⠨.")
            lines.append("- Numeric mode ends when a non-digit appears.\n")
        if "-" in used:
            lines.append("Nemeth minus rules:")
            lines.append("- '-' between numbers → binary subtraction.")
            lines.append("- '-' before a variable → unary minus.\n")
    else:
        if any(ch.isdigit() for ch in used):
            lines.append("UEB numeric rules:")
            lines.append("- Numeric mode begins with ⠼ and continues through digits.\n")
        if "-" in used:
            lines.append("UEB minus rules:")
            lines.append("- '-' is ⠤.\n")
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
    items: list[str],
    model: str = "text-embedding-3-large",
    chunk_size: int = 1000
) -> list[list[float]]:
    """
    Compute embeddings for a large list of strings in safe chunks.
    Returns embeddings in the same order as `items`.
    """

    all_embeddings: list[list[float]] = []

    for start in range(0, len(items), chunk_size):
        end = start + chunk_size
        chunk = items[start:end]

        resp = client.embeddings.create(
            model=model,
            input=chunk
        )

        # Append embeddings in order
        for item in resp.data:
            all_embeddings.append(item.embedding)

    return all_embeddings


def get_or_compute_embeddings(
    examples: list[MathExample],
    cache_path: str,
    use_mathml: bool
) -> list[list[float]]:
    current_hash = hash_examples(examples)

    try:
        stored_hash, stored_embeddings = load_embeddings(cache_path)
        if stored_hash == current_hash:
            logging.info(f"Loaded cached embeddings: {cache_path}")
            return stored_embeddings
        else:
            logging.info(f"Embedding cache invalid: {cache_path}")
    except FileNotFoundError:
        logging.info(f"No embedding cache found: {cache_path}")

    texts = [ex.mathml if use_mathml else ex.braille for ex in examples]
    embeddings = compute_embeddings(items=texts)
    save_embeddings(cache_path, embeddings, current_hash)
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


def choose_k_values(used: set[str], notes: list[str]) -> tuple[int, int]:
    c = estimate_expression_complexity(used, notes)
    if c == "simple":
        return 6, 15
    if c == "medium":
        return 10, 25
    return 16, 35


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
            f"Use the examples to infer the correct mapping from MathML to {code.value.upper()} Braille.\n\n"
        )
        examples_text = "".join(
            f"MathML:\n{examples[i].mathml}\n"
            f"Braille ({code.value}):\n{examples[i].braille}\n\n"
            for i in example_indices
        )
        query_block = (
            f"Now translate the following MathML into {code.value.upper()} Braille.\n"
            f"MathML:\n{query_input}\n\n"
            "Return ONLY the braille characters."
        )
    else:
        header = (
            f"You are a expert {code.value.upper()} Braille to MathML translator.\n"
            f"Use the examples to infer the correct mapping from {code.value.upper()} Braille to MathML.\n"
            "Make sure every example is valid and well-formed MathML.\n\n"
        )
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

    return (
        header
        + examples_text
        + symbol_block
        + context_rules_block
        + structural_block
        + query_block
    )


def build_fallback_prompt(
    query_input: str,
    code: BrailleCode,
    symbol_block: str,
    context_rules_block: str,
    structural_block: str,
    generate_braille: bool
) -> str:

    if generate_braille:
        header = (
            "You are a MathML→Braille translator.\n"
            f"Translate the MathML into {code.value.upper()} Braille.\n"
            "Use standard rules for that code.\n\n"
        )
        body = f"MathML:\n{query_input}\n\nReturn ONLY the braille characters."
    else:
        header = (
            "You are a Braille→MathML translator.\n"
            f"Translate the {code.value.upper()} Braille into MathML.\n"
            "Use standard rules for that code.\n\n"
        )
        body = f"Braille:\n{query_input}\n\nReturn ONLY the MathML markup."

    return header + symbol_block + context_rules_block + structural_block + body


# ============================================================
# GPT Call
# ============================================================
# One shared pool for all GPT calls
_GPT_POOL = ThreadPoolExecutor(max_workers=8)


def _do_gpt_call(
    client: Any,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> tuple[str, int, int, int]:
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
    )
    text = resp.output_text
    usage = resp.usage
    return text, usage.input_tokens, usage.output_tokens, usage.total_tokens


def call_gpt_with_retry(
    client: Any,
    messages: list[dict[str, str]],
    model: str = "gpt-5.2",
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    timeout: float = 30.0,
) -> tuple[str, int, int, int]:
    for attempt in range(max_retries):
        future = _GPT_POOL.submit(_do_gpt_call, client, messages, model, temperature)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            print("GPT call timed out")
        except Exception as e:
            print(f"GPT call failed: {e}")

        if attempt == max_retries - 1:
            raise
        time.sleep(retry_delay)


# ============================================================
# Unified Translation Core
# ============================================================

def translate(
    client: any,
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
    similarity_threshold: float
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
        structural_notes = extract_structural_context(query_input)
        all_embeddings = mathml_embeddings
        overlap_mode: Literal["mathml", "braille", "none"] = "mathml"
    else:
        used_symbols = extract_symbols_from_braille(query_input)
        structural_notes = []
        all_embeddings = braille_embeddings
        overlap_mode = "braille"

    symbol_block = build_symbol_block(used_symbols, symbol_mappings, code, generate_braille)
    context_rules_block = build_context_rules_block(used_symbols, code) if generate_braille else ""
    structural_block = build_structural_context_block(structural_notes) if generate_braille else ""

    k_primary, k_fallback = choose_k_values(used_symbols, structural_notes)
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
        return call_gpt_with_retry(client,prompt)

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
        fb_output = call_gpt_with_retry(client,prompt_fb)
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

    final_prompt = build_fallback_prompt(
        query_input,
        code,
        symbol_block,
        context_rules_block,
        structural_block,
        generate_braille
    )
    return call_gpt_with_retry(client,final_prompt)


# ============================================================
# Async wrappers + unified parallel batch
# ============================================================
async def translate_single_async(
    client: any,
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
    timeout: float = 30.0,
) -> tuple[str, int, int, int]:

    loop = asyncio.get_running_loop()

    # Run translate() in executor, but apply timeout at the async layer
    return await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: translate(
                client,
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
            ),
        ),
        timeout=timeout,
    )


async def parallel_batch_translate(
    client: any,
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
                client,
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
                similarity_threshold
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
                          first_test_index: int,
                          last_test_index: int,
                          info: dict[str, int],  # time is in ms
                          output_file: str) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    """
    usage_info = str(info)[1:-1].replace("'", "").replace(": ", "=")
    print(f"Generated {len(computed_output)} outputs. Stats: {usage_info}ms")
    is_mathml_output = expected_output[0].startswith('<math')
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
                f"Using tests {first_test_index}-{last_test_index} of {len(inputs)} tests.\n")

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
            for tests, computed, expected in zip(inputs, computed_output, expected_output):
                try:
                    checked = areCanonicallyEqual(expected, computed)
                except Exception as e:
                    print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                    checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)

                match = "✓" if checked.isEqual else "✗"
                f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        f.write(f"# Matches: {match_count} out of {len(computed_output)}: {(match_count/len(computed_output)*100):.0f}%.")
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


def parse_cli_args() -> tuple[set[str], int | None, int | None]:
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
        help="Run only the first NUM tests"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Skip the first NUM tests before running"
    )

    args = parser.parse_args()
    modes = {m.lower() for m in args.modes}

    if "all" in modes:
        modes = {"to-nemeth", "to-ueb", "from-nemeth", "from-ueb"}

    invalid = modes - VALID_MODES
    if invalid:
        raise ValueError(f"Invalid mode(s): {invalid}")

    return modes, args.tests, args.start


def run_tests(modes: set[str], n_tests: int | None, test_start: int | None) -> None:
    # ---------------------------------------------------------
    # Load examples and tests
    # ---------------------------------------------------------
    examples = load_math_examples_triple(
        "example_data/canonical-mathml.mmls",
        "example_data/nemeth.brls",
        "example_data/ueb.brls",
    )

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
        examples,
        cache_path="example_data/gpt_mathml_embeddings.json",
        use_mathml=True
    )
    braille_embeddings = get_or_compute_embeddings(
        examples,
        cache_path="example_data/gpt_braille_embeddings.json",
        use_mathml=False
    )

    test_examples = math_examples_from_data(test_mathml, test_nemeth, test_ueb)

    if 'to-nemeth' in modes or 'to-ueb' in modes:
        test_mathml_embeddings = get_or_compute_embeddings(
            test_examples,
            cache_path="test_data/gpt_braille_embeddings.json",
            use_mathml=True
        )
    if 'from-nemeth' in modes or 'from-ueb' in modes:
        test_braille_embeddings = get_or_compute_embeddings(
            test_examples,
            cache_path="test_data/gpt_mathml_embeddings.json",
            use_mathml=False
        )

    output_path_suffix = f"gpt-5.2-{test_start}-{test_start+n_tests}-tests.txt"

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
            "output_path": f"to_nemeth_gpt-{output_path_suffix}",
            "evaluate_label": "MathML → Nemeth",
            "result_var": "to_nemeth_results",
        },
        {
            "mode": "to-ueb",
            "inputs": test_mathml,
            "expected_outputs": test_ueb,
            "code": BrailleCode.UEB,
            "generate_braille": True,
            "output_path": f"to_ueb_gpt-{output_path_suffix}",
            "evaluate_label": "MathML → UEB",
            "result_var": "to_ueb_results",
        },
        {
            "mode": "from-nemeth",
            "inputs": test_nemeth,
            "expected_outputs": test_mathml,
            "code": BrailleCode.NEMETH,
            "generate_braille": False,
            "output_path": f"from_nemeth_gpt-{output_path_suffix}",
            "evaluate_label": "Nemeth → MathML",
            "result_var": "from_nemeth_results",
        },
        {
            "mode": "from-ueb",
            "inputs": test_ueb,
            "expected_outputs": test_mathml,
            "code": BrailleCode.UEB,
            "generate_braille": False,
            "output_path": f"from_ueb_gpt-{output_path_suffix}",
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
                    client=client,
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
                    num_workers=8
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
                test_start,
                test_start + n_tests,
                totals,
                config["output_path"]
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
                    client=client,
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


def main() -> None:
    modes, test_limit, test_start = parse_cli_args()
    run_tests(modes, test_limit, test_start)


if __name__ == "__main__":
    main()
