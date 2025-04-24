import random
from datasets import load_dataset
import re
import string

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
    def qa_prompt(example):
        context = example['context']
        question = example['question']

        prompt_template = (
            "You are a question answering assistant. Given the context, answer the question. "
            "If the answer isn't in the context, respond 'I don't know'.\n\n"

            "Here is an example:\n"
            "Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni)...\n"
            "Question: What is the name of the region the Normans gave their name to?\n"
            "Answer: Normandy\n\n"

            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        return prompt_template.format(context=context, question=question)

    def clean_prediction(self, prediction):
        """
        Cleans the raw prediction output from llama.cpp.
        - Truncates at a new line, 'Context:', or other stop signals.
        - Normalizes the prediction.
        """
        # Split on common stop sequences
        stop_tokens = ["\n\n", "\nContext:", "Context:", "\nQuestion:", "Question", "\nAnswer:", "Answer:"]
        for stop in stop_tokens:
            if stop in prediction:
                prediction = prediction.split(stop)[-1].strip()

        return prediction

    @staticmethod
    def normalize_answer(s):
        """Lowercase, remove punctuation, articles, and normalize whitespace."""
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)  # remove articles
        s = s.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        s = re.sub(r'\s+', ' ', s)  # collapse multiple spaces
        return s.strip()


    def compute_exact_match(self, prediction, ground_truths):
        """Exact match: 1 if prediction is in ground_truths, else 0."""
        prediction = self.normalize_answer(self.clean_prediction(prediction))
        ground_truths = [self.normalize_answer(gt) for gt in ground_truths]

        return int(prediction in ground_truths)


    def compute_f1(self, prediction, ground_truths):
        """Compute the maximum F1 over all ground truths."""
        def get_tokens(s):
            return self.normalize_answer(s).split()

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
        generated = self.clean_prediction(generated)
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
