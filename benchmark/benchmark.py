import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown,
    nvmlDeviceGetUtilizationRates,
)
import time
import pandas as pd
import os
from huggingface_hub import hf_hub_download
import json
import numpy as np
import math
from llama_cpp import Llama
from vllm import LLM, SamplingParams
import evaluate
import re
import string
from sqlglot import parse_one, errors as sqlglot_errors
import torch


class ModelBenchmark:
    def __init__(
        self,
        tokenizer=None,
        max_tokens: int = 256,
        backend="huggingface",
        task="summarization",
        model_path=None,
        llama_gpu_layers=-1,
        verbose=False
    ):
        """
        Initialize benchmarking for Hugging Face, vLLM, and llama.cpp models.

        :param model: Hugging Face/vLLM model or None for llama.cpp
        :param tokenizer: HF tokenizer (optional if using llama.cpp)
        :param max_tokens: Max tokens to generate
        :param backend: 'huggingface', 'vllm', or 'llama.cpp'
        :param llama_model_path: GGUF model path (required if backend == llama.cpp)
        :param llama_gpu_layers: Number of layers to offload to GPU (llama.cpp only)
        """
        self.model_size = self.get_model_size_mb(model_path) if model_path else 0
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.backend = backend
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        # llama.cpp initialization
        if self.backend == "llama.cpp":
            if model_path is None:
                raise ValueError("You must provide llama_model_path for llama.cpp backend.")
            if self.verbose:
                print(f"Loading llama.cpp model from {model_path} ...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=llama_gpu_layers,
                verbose=False
            )
        elif self.backend == "vllm":
            if model_path is None:
                raise ValueError("You must provide model_path for vLLM backend.")
            if self.verbose:
                print(f"Loading vLLM model from {model_path} ...")
            self.llm = LLM(model=model_path)
        else:
            self.llm = None

        # GPU power/memory tracking
        if torch.cuda.is_available():
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

        if self.task == "summarization":
            self.rouge = evaluate.load("rouge")
            self.bertscore = evaluate.load("bertscore")

    def _get_gpu_memory_usage(self):
        """Get GPU memory usage in MB."""
        if self.device == "cuda" and self.handle:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return round(info.used / (1024 ** 2), 2)
        return 0

    def _get_gpu_power_usage(self):
        """Get GPU power usage in Watts."""
        if self.device == "cuda" and self.handle:
            info = nvmlDeviceGetPowerUsage(self.handle)
            return round(info / 1000.0, 2)
        return 0

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage."""
        if self.device == "cuda" and self.handle:
            info = nvmlDeviceGetUtilizationRates(self.handle)
            return round(info.gpu, 2)


    def close(self):
        """Shutdown NVML (clean-up)."""
        if self.device == "cuda":
            nvmlShutdown()

    def measure_ttft(self):
        """
        Measure Time-To-First-Token (TTFT) for a single 50 tokens prompt.
        Uses max_new_tokens=1 and times the full generation call.
        
        Returns:
            dict with TTFT (s), and prompt token count.
        """

        prompt = "Artificial intelligence is a rapidly evolving field with " \
        "applications in healthcare, finance, education, and more. One of the " \
        "most transformative technologies is"

        if self.backend == "vllm":

            sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

            start = time.time()
            _ = self.llm.generate(prompt, sampling_params)
            end = time.time()

        elif self.backend == "llama.cpp":
            start = time.time()
            _ = self.llm(prompt=prompt, temperature=0.0, max_tokens=1)
            end = time.time()

        else:
            raise ValueError(f"Unsupported backend '{self.backend}'")

        return end - start
    

    @staticmethod
    def get_model_size_mb(path):
        """
        Calculate model size in MB. Requires the model path.
        - If `path` is a GGUF file, return its size.
        - If `path` is a directory, sum safetensors, json, tokenizer, etc.
        """
        if os.path.isfile(path) and path.endswith(".gguf"):
            return os.path.getsize(path) / (1024 * 1024)

        elif os.path.isdir(path):
            extensions = (".safetensors", ".bin", ".json", ".txt", ".model", ".tokenizer")
            total_size = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    if f.endswith(extensions):
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
            return total_size / (1024 * 1024)

        else:
            raise ValueError("Path must be a .gguf file or a model directory.")


    def measure_storage(self):
        """Estimate memory + KV cache + model size for HuggingFace (optional for llama.cpp)."""
        mem_usage = self._get_gpu_memory_usage()
        model_size = self.model_size

        return {
            "Memory Usage (MB)": mem_usage,
            "Model Size (MB)": model_size,
            "Overhead (MB)": mem_usage - model_size if mem_usage > model_size else 0,
        }

    def measure_energy(self, num_tokens, num_sentences, generation_time):
        """Compute energy consumption based on time/power."""
        power_watts = self._get_gpu_power_usage()
        total_energy_wh = (power_watts * generation_time) / 3600

        energy_per_token = (
            total_energy_wh * 3600 / num_tokens if num_tokens > 0 else 0
        )
        energy_per_sentence = (
            total_energy_wh * 3600 / num_sentences if num_sentences > 0 else 0
        )

        return {
            "Total Energy (Wh)": round(total_energy_wh, 6),
            "Energy per Token (J/token)": round(energy_per_token, 6),
            "Energy per Sentence (J/sentence)": round(energy_per_sentence, 6),
            "Energy per Second (W)": round(power_watts, 6)
        }

    def measure_gpu_utilization(self):
        """Measure GPU utilization percentage."""
        if self.device == "cuda" and self.handle:
            utilization = self._get_gpu_utilization()
            return {
                "GPU_Utilization (%)": utilization,
            }
        return {}


    def generate_single(self, prompt: str):
        """Generates text and measures generation time."""
        start_time = time.time()

        if self.backend == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.backend == "vllm":
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=0.9,
                top_k=50,
                max_tokens=self.max_tokens,
                stop=["\n"]
            )
            outputs = self.llm.generate(prompt, sampling_params)
            generated_text = outputs[0].outputs[0].text

        elif self.backend == "llama.cpp":
            response = self.llm(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.0,
                stop=["</s>"]
            )
            generated_text = response["choices"][0]["text"].strip()

        else:
            raise ValueError(f"Unsupported backend '{self.backend}'")

        end_time = time.time()
        generation_time = end_time - start_time

        if self.verbose:
            print(f"\nPrompt:\n{prompt}\n\nAnswer:\n{generated_text}\n")
        return generated_text, generation_time
    

    @staticmethod
    def normalize_answer(s):
        """Lowercase, remove punctuation, articles, and normalize whitespace."""
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)  # remove articles
        s = s.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        s = re.sub(r'\s+', ' ', s)  # collapse multiple spaces
        return s.strip()


    def clean_prediction(self, prediction):
        """
        Cleans the raw prediction output from llama.cpp.
        - Truncates at a new line, 'Context:', or other stop signals.
        - Normalizes the prediction.
        """
        # Split on common stop sequences
        stop_tokens = ["\n\n", "\nContext:", "Context:", "\nQuestion:", "SQL:", "\nSQL", "\nAnswer:", "Answer:"]
        for stop in stop_tokens:
            if stop in prediction:
                prediction = prediction.split(stop)[0]

        return prediction


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


    def normalized_equal(self, sql1: str, sql2: str) -> int:
        """
        Compare two SQL strings after normalization (ignores case/spacing).
        """
        return int(self.normalize_answer(sql1) == self.normalize_answer(sql2))

    @staticmethod
    def ast_equal(sql1: str, sql2: str) -> int:
        """
        Compare two SQL statements by parsing them into ASTs.
        Returns True if their structure and logic are equal.
        """
        try:
            tree1 = parse_one(sql1.lower()) # lower, since the reference is lower, even though the columns and tables are not
            tree2 = parse_one(sql2.lower())
            return int(tree1 == tree2)
        except sqlglot_errors.ParseError as e:
            print(f"[AST Parse Error] {e}")
            return int(False)
    

    def _quality_metrics(self, generated: str, reference: str) -> dict:
        """
        Compute evaluation metrics for a single prediction-reference pair based on task type.

        Args:
            generated (str): The generated answer.
            reference (str or list): The ground truth answer(s).
            task (str): Task type, one of ['summarization', 'qa', 'sql'].

        Returns:
            dict: Dictionary of evaluation metric scores.
        """
        if self.task == "summarization":

            generated = generated.strip().split('\n')[0]
            rouge1 = self.rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)["rouge1"]
            rougeL = self.rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)["rougeL"]
            bert = self.bertscore.compute(predictions=[generated], references=[reference], lang="en")["f1"][0]
            return {
                "ROUGE-1": rouge1,
                "ROUGE-L": rougeL,
                "ROUGE_avg": (rouge1 + rougeL) / 2,
                "BERTScore": bert
            }

        elif self.task == "qa":

            generated = self.clean_prediction(generated)
            ref_list = reference if isinstance(reference, list) else [reference]

            em = self.compute_exact_match(generated, ref_list)
            f1 = self.compute_f1(generated, ref_list)

            return {
                "exact_match": em,
                "F1_score": f1
            }

        elif self.task == "sql":

            ast = self.ast_equal(generated, reference)
            normalized_equ = self.normalized_equal(generated, reference)

            return {
                "AST_equal": ast,
                "Normalized_equal": normalized_equ
            }

        else:
            raise ValueError(f"Unsupported task type: {self.task}")



    def benchmark(self, prompts, references):
        results = []

        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"\nEvaluating prompt ({len(prompt)} characters)...")

            # Step 1: Generate text and measure generation time
            generated_text, generation_time = self.generate_single(prompt)

            generated_text = self.clean_prediction(generated_text)

            # Step 2: Count tokens and sentences
            if self.backend == "huggingface":
                num_tokens = len(self.tokenizer(generated_text).input_ids)
            elif self.backend in ["vllm", "llama.cpp"]:
                num_tokens = len(generated_text.split())
            else:
                num_tokens = 0

            num_sentences = generated_text.count('.') + 1

            # Step 3: Latency metrics
            TTFT = self.measure_ttft()
            ATL = generation_time / (num_tokens - 1) if num_tokens > 1 else 0

            latency = {
                "TTFT": round(TTFT, 4),
                "ATL": round(ATL, 4),
                "GL": round(generation_time, 4)
            }

            # Step 4: Throughput
            TPS = num_tokens / generation_time if generation_time > 0 else 0
            SPS = num_sentences / generation_time if generation_time > 0 else 0

            throughput = {
                "TPS": round(TPS, 2),
                "SPS": round(SPS, 2)
            }

            # Step 5: Storage & energy
            storage = self.measure_storage()
            gpu_utilization = self.measure_gpu_utilization()
            energy = self.measure_energy(num_tokens, num_sentences, generation_time)

            # Step 6: Quality metrics
            reference = references[i] if isinstance(references, list) else references
            quality_metrics = self._quality_metrics(generated_text, reference)

            # Step 7: Collect results
            result = {
                "prompt_length": len(prompt),
                "prompt": prompt,
                "generated_answer": generated_text,
                "reference_answer": reference,
                **latency,
                **throughput,
                **storage,
                **gpu_utilization,
                **energy
            }

            if self.task == "sql":
                result["AST_equal"] = quality_metrics["AST_equal"]
                result["Normalized_equal"] = quality_metrics["Normalized_equal"]
            elif self.task == "qa":
                result["exact_match"] = quality_metrics["exact_match"]
                result["F1_score"] = quality_metrics["F1_score"]
            elif self.task == "summarization":
                result["ROUGE-1"] = quality_metrics["ROUGE-1"]
                result["ROUGE-L"] = quality_metrics["ROUGE-L"]
                result["BERTScore"] = quality_metrics["BERTScore"]
            else:
                raise ValueError(f"Unsupported task type: {self.task}")

            results.append(result)

            if self.verbose:
                print(f"\nResults for this prompt:\n{result}")
            
            torch.cuda.empty_cache()

        return pd.DataFrame(results)