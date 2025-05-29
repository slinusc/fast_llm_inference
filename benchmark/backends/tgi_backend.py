import time
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark.backends.base import BaseBackend

# Updated TGIBackend to exactly mirror the provided Docker CLI command
import time
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark.backends.base import BaseBackend

class TGIBackend(BaseBackend):
    """Backend for Hugging Face Text-Generation-Inference via Docker container."""
    def __init__(
        self,
        model_path: str,
        quantization=None,
        max_tokens: int = 2048,
        verbose: bool = False,
        cache_dir: str = "/home/ubuntu/tgi_cache",
        port: int = 8080,
        dtype: str = "float16",
        rope_scaling: str = "dynamic",
        rope_factor: float = 1.0,
    ):
        super().__init__(model_path, quantization, max_tokens, verbose)
        self.cache_dir = cache_dir
        self.port = port
        self.dtype = dtype
        self.rope_scaling = rope_scaling
        self.rope_factor = rope_factor
        self.container_id = None

    def load_model(self):
        # Launch TGI container matching exact CLI
        cmd = [
            "docker", "run", "--gpus", "all", "--rm", "-it", "-d",
            "-p", f"{self.port}:80",
            "-v", f"{self.cache_dir}:/hf_cache",
            "-e", "HF_HUB_OFFLINE=1",
            "ghcr.io/huggingface/text-generation-inference:1.4.1",
            "--model-id", self.model_path,
            "--huggingface-hub-cache", "/hf_cache",
            "--trust-remote-code",
            "--dtype", self.dtype,
            "--rope-scaling", self.rope_scaling,
            "--rope-factor", str(self.rope_factor),
            "--max-total-tokens", str(self.max_tokens),
        ]
        if self.verbose:
            print("Starting TGI container with:", ' '.join(cmd))
        try:
            self.container_id = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start TGI container: {e.output.decode()}")
        self._wait_ready()

    def _wait_ready(self, timeout: int = 90):
        endpoints = ["/healthz", "/health", "/"]
        start = time.time()
        while time.time() - start < timeout:
            for ep in endpoints:
                try:
                    r = requests.get(f"http://localhost:{self.port}{ep}")
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
            time.sleep(1)
        raise TimeoutError("TGI service did not become ready")

    def _single_generate(self, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_path,
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_new_tokens, "do_sample": False}
        }
        resp = requests.post(f"http://localhost:{self.port}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("generated_text") or (data.get("data") or [{}])[0].get("generated_text", "")).strip()

    def generate(self, prompts, task_type=None):
        max_map = {"qa": 32, "sql": 64, "summarization": 256}
        max_new = max_map.get(task_type, self.max_tokens)
        if isinstance(prompts, list):
            results = [None] * len(prompts)
            with ThreadPoolExecutor(max_workers=len(prompts)) as exe:
                futs = {exe.submit(self._single_generate, p, max_new): i for i, p in enumerate(prompts)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        results[idx] = f"<ERROR: {e}>"
            return results
        return self._single_generate(prompts, max_new)

    def measure_ttft(self):
        start = time.time()
        _ = self._single_generate("Test prompt", max_new_tokens=1)
        return time.time() - start

    def close(self):
        if self.container_id:
            subprocess.run(["docker", "stop", self.container_id], check=False)
            if self.verbose:
                print(f"Stopped container {self.container_id}")
            self.container_id = None

    def __del__(self):
        self.close()
