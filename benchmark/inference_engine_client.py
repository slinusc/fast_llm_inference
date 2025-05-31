import os
import subprocess
import time
import requests

class InferenceEngineClient:
    """
    Wrapper for an OpenAI‐compatible server. 
    launch() will call your existing launch_engine.sh script (which runs Docker in the foreground),
    then poll /v1/completions every second until it returns 200.
    """

    def __init__(self, base_url="http://localhost:23333/v1", api_key="none"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = "mistralai/Mistral-7B-Instruct-v0.3"
        self._launcher_proc = None

    def launch(self, backend: str, model_path: str, timeout: float = 500.0):
        """
        1) Starts your existing launch_engine.sh in a Popen (non-blocking).
        2) Polls `http://127.0.0.1:23333/v1/completions` every second
           with a minimal request until we get HTTP 200 (or timeout).
        
        :param backend: One of {tgi, vllm, mii, sglang, lmdeploy}
        :param model_path: HF model ID or local path
        :param timeout: Max seconds to wait for HTTP 200 before raising.
        """
        script_path = os.path.join(os.getcwd(), "benchmark/launch_engine.sh")
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Cannot find '{script_path}'.")
        if not os.access(script_path, os.X_OK):
            raise PermissionError(f"'{script_path}' is not executable (chmod +x missing).")

        # 1) Start the launcher in a subprocess.
        #    Because your script does `docker run --rm ...` (no "-d"), it will block in the foreground.
        #    Running it under Popen lets Python continue while Docker is starting.
        cmd = [
            script_path,
            f"--engine={backend}",
            f"--model={model_path}"
        ]
        self._launcher_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 2) Poll /v1/completions until we get 200 or timeout.
        url = "http://127.0.0.1:23333/v1/completions"
        payload = {
            "model": model_path,
            "prompt": "",
            "max_tokens": 1
        }
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            # Check if the launcher process has died unexpectedly:
            if self._launcher_proc.poll() is not None:
                # Exit code != 0 means the docker run failed
                raise RuntimeError(
                    f"Launcher script exited early with code {self._launcher_proc.returncode}"
                )

            try:
                r = requests.post(url, json=payload, headers=headers, timeout=2.0)
                if r.status_code == 200:
                    # Server is ready!
                    break
            except requests.exceptions.RequestException:
                # Connection refused, timeout, etc. → server not up yet
                pass

            elapsed = time.time() - start_time
            if elapsed > timeout:
                # Abort if we've waited too long
                raise TimeoutError(
                    f"Waited {timeout}s for {url} to return 200, but never saw it. "
                    "Is the container failing to start?"
                )
            time.sleep(2.0)

        # At this point, container is up and /v1/completions returns 200.
        # We simply return; the Docker container keeps running in the foreground.
        return

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
        model = model or self.default_model
        is_batch = isinstance(prompt, (list, tuple))

        resp = self.client.completions.create(
            model=model,
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
        model = self.default_model

        start = time.time()
        stream_resp = self.client.completions.create(
            model=model,
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
