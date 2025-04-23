import random
import json
from datasets import load_dataset
from sqlglot import parse_one, errors as sqlglot_errors

class SqlTask:
    """
    A class to handle the SQL generation task using the Spider dataset.
    """

    def __init__(self, num_examples=100, tables_path="/home/ubuntu/fast_llm_inference/tables.json"):
        random.seed(42)
        self.dataset = list(load_dataset("spider", split="test"))
        self.num_examples = num_examples
        self.tables_path = tables_path

    def generate_prompts(self):
        """
        Generates prompts and references for SQL generation using the Spider dataset.
        """
        sampled = random.sample(self.dataset, self.num_examples)
        prompts = [self.sql_prompt(example) for example in sampled]
        references = [example["query"] for example in sampled]
        return prompts, references

    def get_table_columns(self, db_id: str) -> str:
        """
        Load table schema and format as prompt string.
        """
        with open(self.tables_path, "r") as f:
            tables_json = json.load(f)

        for db in tables_json:
            if db["db_id"] == db_id:
                table_names = db["table_names_original"]
                column_info = db["column_names_original"]
                table_columns = {table: [] for table in table_names}
                for table_idx, col_name in column_info:
                    if col_name == "*":
                        continue
                    table = table_names[table_idx]
                    table_columns[table].append(col_name)
                return "\n".join(f"Table '{t}': columns = {', '.join(cols)}" for t, cols in table_columns.items())

        return "No schema found for that db_id."

    def sql_prompt(self, example):
        question = example["question"]
        db_id = example["db_id"]
        column_context = self.get_table_columns(db_id)

        prompt_template = (
            "You are a SQL query generation assistant. Given a natural language question, generate the corresponding SQL query.\n"
            "Only generate valid SQL statements, no explanations or extra text.\n\n"

            "Here is an example:\n\n"
            "Question: How many heads of the departments are older than 56?\n\n"
            "Tables in the database:\n"
            "Table 'department': columns = Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees\n"
            "Table 'head': columns = head_ID, name, born_state, age\n"
            "Table 'management': columns = department_ID, head_ID, temporary_acting\n\n"
            "SQL: SELECT count(*) FROM head WHERE age > 56\n\n"

            "Question: {question}\n\n"
            "Tables in the database:\n"
            "{column_context}\n\n"
            "SQL:"
        )

        return prompt_template.format(question=question, column_context=column_context)

    @staticmethod
    def quality_metrics(generated, reference):

        def normalize_answer(s):
            import re, string
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = re.sub(r'\s+', ' ', s)
            return s.strip()

        def clean_prediction(prediction):
            stop_tokens = ["\n\n", "\nContext:", "Context:", "\nQuestion:", "SQL:", "\nSQL", "\nAnswer:", "Answer:"]
            for stop in stop_tokens:
                if stop in prediction:
                    prediction = prediction.split(stop)[0]
            return prediction

        def ast_equal(sql1, sql2):
            try:
                tree1 = parse_one(sql1.lower())
                tree2 = parse_one(sql2.lower())
                return int(tree1 == tree2)
            except sqlglot_errors.ParseError:
                return 0

        def normalized_equal(sql1, sql2):
            return int(normalize_answer(sql1) == normalize_answer(sql2))

        pred = clean_prediction(generated)

        return {
            "AST_equal": ast_equal(pred, reference),
            "Normalized_equal": normalized_equal(pred, reference)
        }


if __name__ == "__main__":
    task = SqlTask(num_examples=3)
    prompts, references = task.generate_prompts()
    for i in range(3):
        print(f"Prompt {i+1}:")
        print(prompts[i])
        print(f"Reference {i+1}:")
        print(references[i])
        print()
