import random
from datasets import load_dataset
import re
import string
from ..utils import normalize_answer

class QATask:
    """
    A class to handle the question answering task using the SQuAD v2 dataset.
    """

    def __init__(self):
        random.seed(42)
        self.dataset = load_dataset("squad_v2")

    def generate_prompts(self, num_examples : int = 100):
        """
        Generates prompts and references for QA using SQuAD v2.
        """
        validation_list = list(self.dataset["validation"])
        sampled_questions = random.sample(validation_list, 2 * num_examples)

        # Keep only examples that have at least one answer
        questions_with_answers = [ex for ex in sampled_questions if len(ex['answers']['text']) > 0][:num_examples]

        prompts = [self.qa_prompt(example) for example in questions_with_answers]
        references = [example['answers']['text'] for example in questions_with_answers]

        return prompts, references

    @staticmethod
    def qa_prompt(example: dict) -> str:

        # 1. System instruction
        system_message = (
            "You are a question-answering assistant. Answer in exactly **ONE** line. "
            "If the answer is not contained in the context, don't answer. "
            "If it is possible just answer with a single word or a short phrase. "
        )

        # 2. Few-shot demonstration
        demo_block = (
            "### EXAMPLES\n"
            "Context:\n"
            "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni)…\n"
            "Question:\n"
            "What is the name of the region the Normans gave their name to?\n"
            "Answer:\n"
            "Normandy\n\n"
        )

        # 3. Instruction header
        instruction = (
            "### INSTRUCTION\n"
            "Read the context and answer the question.\n\n"
        )

        # 4. Input header
        input_block = (
            "### INPUT\n"
            f"Context:\n{example['context']}\n\n"
            f"Question:\n{example['question']}\n\n"
        )

        # 5. Output header (model writes its answer here)
        output_block = ("### OUTPUT\n"
            "Answer:"
        )

        return (
            f"### SYSTEM\n{system_message}\n\n"
            f"{demo_block}"
            f"{instruction}"
            f"{input_block}"
            f"{output_block}"
        )


    def compute_exact_match(self, prediction, ground_truths):
        """Exact match: 1 if prediction is in ground_truths, else 0."""
        prediction = normalize_answer(prediction)
        ground_truths = [normalize_answer(gt) for gt in ground_truths]

        return int(prediction in ground_truths)


    def compute_f1(self, prediction, ground_truths):
        """Compute the maximum F1 over all ground truths."""
        def get_tokens(s):
            return normalize_answer(s).split()

        pred_tokens = get_tokens(prediction)
        if not pred_tokens:
            return int(not any(get_tokens(gt) for gt in ground_truths))

        scores = []
        for gt in ground_truths:
            gt_tokens = get_tokens(gt)
            common = set(pred_tokens) & set(gt_tokens)
            num_same = len(common)

            if num_same == 0:
                scores.append(0)
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)

        return max(scores)

    def quality_metrics(self, generated, reference):
        ref_list = reference if isinstance(reference, list) else [reference]

        em = self.compute_exact_match(generated, ref_list)
        f1 = self.compute_f1(generated, ref_list)

        return {
            "exact_match": em,
            "F1_score": f1
        }


if __name__ == "__main__":
    task = QATask(num_examples=3)
    prompts, references = task.generate_prompts()
    for i in range(3):
        print(f"Prompt {i+1}:")
        print(prompts[i])
        print(f"Reference {i+1}:")
        print(references[i])
        print()
