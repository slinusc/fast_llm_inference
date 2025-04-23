import random
from datasets import load_dataset

class QATask:
    """
    A class to handle the question answering task using the SQuAD v2 dataset.
    """

    def __init__(self, num_examples=100):
        random.seed(42)
        self.dataset = load_dataset("squad_v2")
        self.num_examples = num_examples

    def generate_prompts(self):
        """
        Generates prompts and references for QA using SQuAD v2.
        """
        validation_list = list(self.dataset["validation"])
        sampled_questions = random.sample(validation_list, 2 * self.num_examples)

        # Keep only examples that have at least one answer
        questions_with_answers = [ex for ex in sampled_questions if len(ex['answers']['text']) > 0][:self.num_examples]

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
