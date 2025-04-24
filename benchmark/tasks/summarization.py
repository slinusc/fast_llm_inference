import random
from datasets import load_dataset
import evaluate

class SummarizationTask:
    """
    A class to handle the summarization task using the CNN/DailyMail dataset.
    """

    def __init__(self):
        random.seed(42)
        self.rouge = evaluate.load("rouge")
        self.dataset = list(load_dataset("cnn_dailymail", "3.0.0")["test"])

    def generate_prompts(self, num_examples : int = 100):
        """
        Generates random summarization prompts and references.
        """
        sampled = random.sample(self.dataset, num_examples)
        prompts = [self.sum_prompt(example["article"]) for example in sampled]
        references = [example["highlights"] for example in sampled]
        return prompts, references

    @staticmethod
    def sum_prompt(article : str) -> str:
        """
        Summarize the given `article` into a 2–3 sentence summary (CNN/DailyMail style), using a single demonstration example.
        """
        prompt = (
            "You are a news summarization assistant. Given a full news article, produce a concise and informative summary in 2–3 sentences.\n\n"

            "Example:\n\n"

            "Article:\n"
            "(CNN) -- The partnership started as a single shop on Oxford Street in London, opened in 1864 by John Lewis. "
            "Today the partnership is an organization with bases throughout the UK, with supermarkets and department stores, "
            "employing approximately 67,100 people. All 67,100 permanent staff are Partners who own 26 John Lewis department stores, "
            "183 Waitrose supermarkets, an online and catalogue business, John Lewis Direct a direct services company - Greenbee, "
            "three production units and a farm. Every Partner receives the same scale of bonus, based on a fixed percentage of their annual wage. "
            "The bonus for 2006 was 18% equivalent to 9 weeks pay, which was rolled out for every employee. "
            "Chairman Sir Stuart Hampson retired at the end of March 2007, his successor is Charlie Mayfield. Hampson's salary for January 26, "
            "2006 to January 26, 2007 was $1.66 million which included the partnership bonus of $250,000. John Lewis' consolidated revenue for "
            "the last financial year was $11.4 billion.  E-mail to a friend ."

            "Summary:\n"
            "John Lewis Partnership began as a shop on London's Oxford street in 1864 .\n"
            "All 67,100 employees are partners in the organization and own shares ."

            "Now summarize the following article:\n\n"

            f"Article: {article}\n"
        
            "Summary:\n"
        )

        return prompt

    @staticmethod
    def clean_prediction(self, prediction):
        """
        Cleans the raw prediction output from llama.cpp.
        - Truncates at a new line, 'Context:', or other stop signals.
        - Normalizes the prediction.
        """
        # Split on common stop sequences
        stop_tokens = ["\nArticle:", "Article:", "\nSummary:", "Summary:"]
        for stop in stop_tokens:
            if stop in prediction:
                prediction = prediction.split(stop)[-1].strip()

        return prediction
    
    def quality_metrics(self, generated : str, reference : str) -> float:
        """
        Returns the metric used for evaluation.
        """
        generated = generated.strip().split('\n')[0]
        rouge1 = self.rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)["rouge1"]
        rouge2 = self.rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)["rouge2"]
        rougeL = self.rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)["rougeL"]
        
        return {
            "ROUGE-1": rouge1,
            "ROUGE-2": rouge2,
            "ROUGE-L": rougeL
        }


if __name__ == "__main__":
    task = SummarizationTask()
    prompts, references = task.generate_prompts(num_examples=3)
    for i in range(3):
        print(f"Prompt {i+1}:\n{prompts[i]}\n")
        print(f"Reference {i+1}:\n{references[i]}\n")
