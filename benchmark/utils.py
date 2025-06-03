import re, string

def tok_cnt_sql(text: str) -> int:
    """
    Count tokens in an SQL query (or multiple statements).
    Returns the number of non‐whitespace, non‐comment tokens after sqlparse.lexing.
    """
    # Parse into one or more Statement objects
    parsed = sqlparse.parse(text)
    if not parsed:
        return 0

    # We’ll just tokenize every statement in the tuple and flatten them
    count = 0
    for stmt in parsed:
        # stmt.tokens is a TokenList; flatten() yields every subtoken
        for tk in stmt.flatten():
            # Skip pure whitespace or comments
            if tk.ttype is Whitespace or tk.ttype is Comment:
                continue
            # Everything else (Keyword, Name, Number, Punctuation, Operator, etc.) counts
            count += 1
    return count


def sent_cnt_sql(text: str) -> int:
    """
    Count “statements” in an SQL string by asking sqlparse to split into Statement objects.
    Each semicolon (or run of semicolons) produces a new Statement in sqlparse.parse().
    """
    parsed = sqlparse.parse(text)
    # Filter out any empty statements that contain only whitespace
    nonempty = [stmt for stmt in parsed if not stmt.is_whitespace]
    return len(nonempty)

    

def tok_cnt(text: str, mode: str = "qa") -> int:
    """
    Count tokens in `text`.
      • mode="sql": identifiers or single-char punctuation (for SQL).
    """
    if mode == "sql":
        # matches identifiers/numbers or any of ; ( ) , = * < >
        return tok_cnt_sql(text)
    else:
        return len(text.split())


# ── Sentence/Statement Counters ──

def sent_cnt(text: str, mode: str = "qa") -> int:
    """
    Count “sentences” in `text`.
      • mode="sql": count SQL statements separated by semicolons.
    """
    if mode == "sql":
        # count semicolons (each ';' marks end of a statement)
        return text.count(";")
    else:
        # count runs of . ! ? or ellipsis
        return len(re.findall(r"[.!?…]+", text))


def chunker(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

def normalize_answer(s: str) -> str:
            
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = re.sub(r'\s+', ' ', s)
            return s.strip()

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

