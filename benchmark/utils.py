import re, string
import transformers
from typing import Dict

# Cache tokenizers so repeated calls with the same model_name don’t reload each time
_tokenizer_cache: Dict[str, transformers.PreTrainedTokenizerFast] = {}

def tok_cnt(text: str, model_name: str) -> int:
    """
    Returns the number of tokens in `text` according to the tokenizer
    for the given `model_name`. Tokenization always runs on CPU.
    """
    # Load (or retrieve from cache) the tokenizer for this model
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
    tokenizer = _tokenizer_cache[model_name]

    # Tokenize without adding special tokens and return length
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


# ── Sentence/Statement Counters ──

def sent_cnt(text: str, mode: str = "qa") -> int:
    """
    Count “sentences” in `text`.
      • mode="sql": count SQL statements separated by semicolons.
      • mode="qa" (default): count punctuation-based sentence boundaries.
    Returns at least 1.
    """
    if mode == "sql":
        # In SQL mode, assume minimum 1 statement even if no semicolon
        return 1
    else:
        count = len(re.findall(r"[.!?…]+", text))
        return max(count, 1)


def chunker(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

def normalize_answer(s: str) -> str:
    """
    Lowercase, remove punctuation/articles, collapse whitespace.
    Handles None inputs gracefully.
    """
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def clean_prediction(prediction: list[str]) -> list[str]:
    cleaned = []
    for raw in prediction:
        # 1) Remove anything after the first '###'
        ans = raw.split("###", 1)[0]

        # 2) Strip whitespace (including newlines) from both ends
        ans = ans.strip()

        # 3) Remove anything after the first newline (in the stripped string)
        ans = ans.split("\n", 1)[0]

        cleaned.append(ans)
    return cleaned

