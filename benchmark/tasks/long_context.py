import random
import re
import string
from typing import Dict, List, Tuple
from datasets import load_dataset
from ..utils import normalize_answer

class LongContextQATask:
    """
    Given the “slinusc/ContextStretchQA” dataset, this Task lets you sample
    a specified number of examples from each context_range bucket
    ("3k", "4k", "8k", "16k", "32k"). The generate_prompts(...) method takes
      - num_samples_per_level: how many examples to return from each bucket
    and returns a dict mapping each range to its (prompts, references) pair:

        {
            "3k":  (prompts_3k, refs_3k),
            "4k":  (prompts_4k, refs_4k),
            "8k":  (prompts_8k, refs_8k),
            "16k": (prompts_16k, refs_16k),
            "32k": (prompts_32k, refs_32k),
        }

    Raises ValueError if any bucket has fewer examples than requested.
    """

    def __init__(self):
        random.seed(42)

        # 1) Load the HF dataset (train split)
        ds = load_dataset("slinusc/ContextStretchQA")["train"]

        # 2) Build a mapping from context_range → list of examples
        #    Each example is a dict with keys "context", "question", "answer", "length".
        self.by_range: Dict[str, List[Dict[str, str]]] = {}
        for row in ds:
            cr = row["context_range"]
            if cr not in self.by_range:
                self.by_range[cr] = []
            self.by_range[cr].append({
                "context": row["context"],
                "question": row["question"],
                "answer": row["answer"],
                "length": row["length"],
            })

        # Validate that the expected five ranges exist
        expected = {"3k", "4k", "8k", "16k", "32k"}
        missing = expected - set(self.by_range.keys())
        if missing:
            raise ValueError(f"Missing context_range classes: {missing}")

    def generate_prompts(
        self,
        num_samples_per_level: int
    ) -> Tuple[List[str], List[str], List[int], List[str]]:
        """
        For each context_range level ("3k", "4k", "8k", "16k", "32k"),
        sample exactly `num_samples_per_level` examples (without replacement).

        Returns four parallel lists:
        - prompts:    concatenation of prompts from each range
        - references: ground-truth answers
        - lengths:    original context lengths (in tokens)
        - ranges:     corresponding context_range label

        Raises ValueError if any level has too few examples.
        """
        prompts: List[str] = []
        refs: List[str] = []
        lengths: List[int] = []
        crs: List[str] = []

        for cr, examples in self.by_range.items():
            if len(examples) < num_samples_per_level:
                raise ValueError(
                    f"Not enough examples in context_range='{cr}'. "
                    f"Requested {num_samples_per_level}, but only {len(examples)} available."
                )

            sampled = random.sample(examples, num_samples_per_level)

            for ex in sampled:
                prompts.append(self._build_prompt(ex))
                refs.append(ex["answer"])
                lengths.append(ex["length"])
                crs.append(cr)

        return prompts, refs, lengths, crs



    @staticmethod
    def _build_prompt(example: Dict[str, str]) -> str:
        """
        Construct the full prompt string for a single example:
          - SYSTEM instruction
          - Few-shot demonstration
          - INSTRUCTION header
          - INPUT header (context + question)
          - OUTPUT header
        """
        system_message = (
            "You are a question-answering assistant. Answer in exactly ONE line. "
            "If the answer is not contained in the context, answer 'unanswerable'. "
            "If it's a yes/no question, respond with 'yes' or 'no'.\n\n"
        )

        demo_block = (
            "### EXAMPLES\n"
            "Context:\n"
            "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni)…\n"
            "Question:\n"
            "What is the name of the region the Normans gave their name to?\n"
            "Answer:\n"
            "Normandy\n\n"
        )

        instruction = (
            "### INSTRUCTION\n"
            "Read the context and answer the question.\n\n"
        )

        input_block = (
            "### INPUT\n"
            f"Context:\n{example['context']}\n\n"
            f"Question:\n{example['question']}\n\n"
        )

        output_block = "### OUTPUT\nAnswer:"

        return (
            f"### SYSTEM\n{system_message}"
            f"{demo_block}"
            f"{instruction}"
            f"{input_block}"
            f"{output_block}"
        )

    def compute_exact_match(self, prediction: str, ground_truths: List[str]) -> int:
        """
        Returns 1 if normalized(prediction) is in normalized(ground_truths), else 0.
        """
        pred = normalize_answer(prediction)
        gts = [normalize_answer(gt) for gt in ground_truths]
        return int(pred in gts)

    def compute_f1(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute maximum F1 among all ground truths.
        """
        def _tokens(text: str) -> List[str]:
            return normalize_answer(text).split()

        pred_tokens = _tokens(prediction)
        if not pred_tokens:
            return int(not any(_tokens(gt) for gt in ground_truths))

        scores: List[float] = []
        for gt in ground_truths:
            gt_tokens = _tokens(gt)
            common = set(pred_tokens) & set(gt_tokens)
            if not common:
                scores.append(0.0)
                continue
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)
        return max(scores)

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Return a dict with:
          - "exact_match": 0 or 1
          - "F1_score": float
        """
        return {
            "exact_match": self.compute_exact_match(generated, [reference]),
            "F1_score": self.compute_f1(generated, [reference]),
        }


# Example usage:
if __name__ == "__main__":
    task = LongContextQATask()
    batches = task.generate_prompts(num_samples_per_level=100)
    for cr, (prompts, refs) in batches.items():
        print(f"Context Range {cr}: {len(prompts)} prompts, {len(refs)} references")
        print(f"First prompt (truncated):\n{prompts[0][:200]}...\n")
        print(f"First reference: {refs[0]}\n")
