import os
import subprocess
import time
import requests
from pathlib import Path

class InferenceEngineClient:
    """
    Wrapper for an OpenAI‐compatible server. 
    launch() will call your existing launch_engine.sh script (which runs Docker in the foreground),
    then poll /v1/completions every second until it returns 200.
    """

    def __init__(self, base_url="http://localhost:23333/v1", api_key="none"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._launcher_proc = None
        self.model = None

    

    def launch(self, backend: str, model: str, timeout: float = 500.0):
        """
        1) Starts your existing launch_engine.sh in a Popen (non-blocking).
        2) Polls http://127.0.0.1:23333/v1/models every 2 s until <model> appears
        or ‘timeout’ seconds elapse.

        :param backend: One of {tgi, vllm, mii, sglang, lmdeploy}
        :param model:   HF model ID or local path
        :param timeout: Seconds to wait for the model to show up before raising
        """

        # ── 1. Docker-image guard ────────────────────────────────────────────
        _image = {
            "tgi":      "ghcr.io/huggingface/text-generation-inference:latest",
            "vllm":     "docker.io/vllm/vllm-openai:latest",
            "mii":      "microsoft/deepspeed-mii:latest",
            "sglang":   "docker.io/lmsysorg/sglang:latest",
            "lmdeploy": "docker.io/openmmlab/lmdeploy:latest",
        }.get(backend)

        if _image is None:
            raise ValueError(f"Unknown backend '{backend}'")

        image_present = subprocess.run(
            ["docker", "inspect", "--type=image", _image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

        if not image_present:
            print(f"[launch] pulling Docker image {_image} …")
            try:
                subprocess.check_call(["docker", "pull", _image])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to pull {_image}: {e}") from e
            print("[launch] image ready")
        # ──────────────────────────────────────────────────────────────────────

        # ── 2. Sanity-check launch_engine.sh path / permissions ──────────────
        script_path = (Path(__file__).resolve().parent / "launch_engine.sh").as_posix()
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Cannot find '{script_path}'.")
        if not os.access(script_path, os.X_OK):
            raise PermissionError(f"'{script_path}' is not executable (chmod +x missing).")
        # ──────────────────────────────────────────────────────────────────────

        # ── 3. Start the launcher in a subprocess (non-blocking) ─────────────
        cmd = [
            script_path,
            f"--engine={backend}",
            f"--model={model}",
        ]
        self._launcher_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # ──────────────────────────────────────────────────────────────────────

        # ── 4. Poll /v1/models until the model appears or we timeout ─────────
        list_url   = "http://127.0.0.1:23333/v1/models"
        start_time = time.time()

        while True:
            # if the launcher died early, surface the error
            if self._launcher_proc.poll() is not None:
                raise RuntimeError(
                    f"Launcher script exited early with code {self._launcher_proc.returncode}"
                )

            try:
                resp = requests.get(list_url, timeout=2.0)
                if resp.status_code == 200:
                    for entry in resp.json().get("data", []):
                        if entry.get("id") == model:
                            self.model = model             # model is ready
                            return
            except requests.exceptions.RequestException:
                pass  # server not up yet → keep polling

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Waited {timeout}s for model '{model}' at {list_url}, but it never appeared."
                )
            time.sleep(2.0)



    def completion(
        self,
        prompt,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_p: float = 0.9,
        stream: bool = False,
    ):
        """
        Send one or more prompts. :param prompt: string or list[str]
        """
        model = model
        is_batch = isinstance(prompt, (list, tuple))

        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
        )

        if stream:
            return resp

        texts = [c.text for c in resp.choices]
        return texts if is_batch else texts[0]

    def warmup(self, num_iters: int = 3):
        """
        Send a few small dummy requests to load the model into memory and
        JIT any kernels so that subsequent inference calls are faster.
        :param model: HF model ID or local path; defaults to self.default_model
        :param num_iters: Number of warmup calls to make
        """

        dummy_prompt = "Warmup"

        for _ in range(num_iters):
            try:
                _ = self.client.completions.create(
                    model=self.model,
                    prompt=dummy_prompt,
                    max_tokens=1,
                    temperature=0.1,
                )
            except Exception:
                # If the server isn't ready yet, retry after a short sleep
                time.sleep(1.0)
                continue

    def measure_ttft(self) -> float:
        """
        Issue a 1‐token streaming request and measure the time until the first chunk arrives.
        """
        import time
        prompt = (
            "Artificial intelligence is a rapidly evolving field with applications in "
            "healthcare, finance, education, and more. One of the most transformative "
            "technologies is"
        )

        start = time.time()
        stream_resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.1,
            stream=True,
        )

        first_token_time = None
        for chunk in stream_resp:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and delta.get("text", "").strip() != "":
                first_token_time = time.time()
                break

        if first_token_time is None:
            first_token_time = time.time()

        return first_token_time - start

    def close(self):
        """
        Stop any Docker container exposing port 23333, then close the HTTP client.
        """
        # 1) Attempt to find and stop the container(s) on port 23333
        try:
            # This returns all container IDs whose published port includes 23333/tcp
            result = subprocess.run(
                ["docker", "ps", "--filter", "publish=23333", "-q"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True
            )
            container_ids = result.stdout.strip().split()
            for cid in container_ids:
                # Gracefully stop each container
                subprocess.run(
                    ["docker", "stop", cid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        except subprocess.CalledProcessError:
            # If `docker ps` itself fails (e.g. Docker not running), just skip
            pass

        # 2) Now terminate the launcher subprocess if it’s still alive
        if hasattr(self, "_launcher_proc") and self._launcher_proc:
            if self._launcher_proc.poll() is None:
                # Process is still running; give it a gentle terminate
                self._launcher_proc.terminate()
                try:
                    self._launcher_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._launcher_proc.kill()

        # 3) Finally, close any HTTP client connections
        try:
            self.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    ### ttft
    client = InferenceEngineClient()
    client.launch(backend="tgi", model="Qwen/Qwen2.5-7B-Instruct")
    client.warmup()
    ttft = client.measure_ttft()
    print(f"TTFT: {ttft:.3f} seconds")
    client.close()