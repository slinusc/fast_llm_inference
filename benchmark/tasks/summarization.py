import random
from datasets import load_dataset

class SummarizationTask:
    """
    A class to handle the summarization task using the CNN/DailyMail dataset.
    """

    def __init__(self):
        random.seed(42)
        self.dataset = list(load_dataset("cnn_dailymail", "3.0.0")["test"])

    def generate_prompts(self, num_examples=100):
        """
        Generates random summarization prompts and references.
        """
        sampled = random.sample(self.dataset, num_examples)
        prompts = [self.sum_prompt(example["article"]) for example in sampled]
        references = [example["highlights"] for example in sampled]
        return prompts, references

    @staticmethod
    def sum_prompt(article):
        """
        Summarize the given `article` into a 2–3 sentence summary (CNN/DailyMail style), using a single demonstration example.
        """
        prompt = (
            "You are a news summarization assistant. Given a full news article, produce a concise and informative summary in 2–3 sentences.\n\n"

            "Example:\n\n"

            "Article: President Biden held a press conference today addressing the growing concerns over inflation and rising gas prices. "
            "He outlined the administration's plans to release additional oil reserves and invest in clean energy initiatives to ease economic pressures. "
            "The President also reassured the public that steps are being taken to stabilize the supply chain and reduce long-term costs. "
            "Reporters questioned whether these measures would have a short-term impact, but Biden remained confident in the approach.\n\n"

            "Summary: President Biden announced plans to combat inflation by releasing oil reserves and investing in clean energy. "
            "He also pledged to address supply chain issues and stabilize costs.\n\n"

            "Now summarize the following article:\n\n"
            f"Article: {article}\n"
            "Summary:"
        )

        return prompt


if __name__ == "__main__":
    task = SummarizationTask()
    prompts, references = task.generate_prompts(num_examples=3)
    for i in range(3):
        print(f"Prompt {i+1}:\n{prompts[i]}\n")
        print(f"Reference {i+1}:\n{references[i]}\n")