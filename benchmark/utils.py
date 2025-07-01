import re, string
from typing import Dict
from collections import deque
import threading
from threading import Thread
from math import floor


def tok_cnt(text: str, inflation: float = 1.2) -> int:
    """
    Approximate token count using whitespace-based word splitting,
    scaled by a token inflation factor (default 1.2).

    Returns at least 1.
    """
    if not text.strip():
        return 1
    word_count = len([tok for tok in text.split() if tok])
    token_estimate = floor(word_count * inflation)
    return max(token_estimate, 1)


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


def _start_log_tailer(self, max_lines: int = 30):
    """Spawn a daemon thread that reads the process’ output and
    stores the latest `max_lines` in self._log_buf (deque[str])."""
    self._log_buf = deque(maxlen=max_lines)

    def _tail():
        for line in self._launcher_proc.stdout:
            self._log_buf.append(line.rstrip("\n"))
            if self._stop_tail.is_set():
                break

    self._stop_tail = threading.Event()
    self._tail_thread = Thread(target=_tail, daemon=True)
    self._tail_thread.start()
