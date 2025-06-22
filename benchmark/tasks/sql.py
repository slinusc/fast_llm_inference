import random
import json
from datasets import load_dataset
from sqlglot import parse_one, errors as sqlglot_errors

class SQLTask:
    """
    A class to handle the SQL generation task using the Spider dataset.
    """

    def __init__(self, tables_path="/home/ubuntu/fast_llm_inference/benchmark/lookup/tables.json"):
        random.seed(42)
        self.dataset = list(load_dataset("spider", split="train"))
        self.tables_path = tables_path

    def generate_prompts(self, num_examples : int = 100):
        """
        Generates prompts and references for SQL generation using the Spider dataset.
        """
        sampled = random.sample(self.dataset, num_examples)
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

    def sql_prompt(self, example: dict) -> str:

        question = example["question"]
        column_context = self.get_table_columns(example["db_id"])

        # 1. System instruction
        system_message = (
            "You are a SQL query generation assistant. Given a natural language question, "
            "generate the corresponding SQL statement. Only generate valid SQL statements—no "
            "explanations or extra text. Always end the SQL statement with a semicolon.\n\n"
        )

        # 2. Few-shot demonstration
        demo_block = (
            "### EXAMPLES\n"
            "Question:\n"
            "How many heads of the departments are older than 56?\n\n"
            "Tables in the database:\n"
            "Table 'department': columns = Department_ID, Name, Creation, Ranking, "
            "Budget_in_Billions, Num_Employees\n"
            "Table 'head':       columns = head_ID, name, born_state, age\n"
            "Table 'management': columns = department_ID, head_ID, temporary_acting\n\n"
            "SQL:\n"
            "SELECT count(*) FROM head WHERE age > 56\n\n"
        )

        # 3. Instruction header
        instruction = (
            "### INSTRUCTION\n"
            "Generate the SQL statement that answers the question.\n\n"
        )

        # 4. Input header
        input_block = (
            "### INPUT\n"
            f"Question:\n{question}\n\n"
            "Tables in the database:\n"
            f"{column_context}\n\n"
        )

        # 5. Output header + end sentinel
        output_and_end = (
            "### OUTPUT\n"
            "SQL:\n"
        )

        return (
            f"### SYSTEM\n{system_message}\n\n"
            f"{demo_block}"
            f"{instruction}"
            f"{input_block}"
            f"{output_and_end}"
        )


    def quality_metrics(self, generated, reference):

        from ..utils import normalize_answer

        def ast_equal(sql1: str, sql2: str) -> int:
            try:
                tree1 = parse_one(sql1.lower())
                tree2 = parse_one(sql2.lower())
                return int(tree1 == tree2)
            except Exception:  # catches ParseError, TokenError, AttributeError, etc.
                return 0

        def normalized_equal(sql1, sql2):
            return int(normalize_answer(sql1) == normalize_answer(sql2))

        return {
            "AST_equal": ast_equal(generated, reference),
            "Normalized_equal": normalized_equal(generated, reference)
        }


if __name__ == "__main__":
    task = SQLTask(tables_path="/home/rag/fast_llm_inference/benchmark/tasks/tables.json")
    prompts, references = task.generate_prompts(num_examples=3)
    for i in range(3):
        print(f"Prompt {i+1}:")
        print(prompts[i])
        print(f"Reference {i+1}:")
        print(references[i])
        print()
