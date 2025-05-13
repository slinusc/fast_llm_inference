# benchmark/scenarios.py
import asyncio, random, time, math
from statistics import mean
from benchmark.benchmark import ModelBenchmark

class ScenarioRunner:
    def __init__(self, bench_cfg, scenario, qps=1.0, duration_s=300):
        self.bench_cfg = bench_cfg          # dict ⟶ args for ModelBenchmark
        self.scenario  = scenario           # "single", "batch", "server"
        self.qps       = qps                # only for server mode
        self.duration  = duration_s
        self.seeds     = 42

    # ────────────────────────────────────────────────────────
    async def _server_loop(self, mb: ModelBenchmark):
        "Fire prompts so that long-run average == self.qps."
        from benchmark.tasks.qa   import QATask
        task = QATask()                       # pick any; your CLI can override
        prompts, refs = task.generate_prompts(10_000)

        inter_arrival = 1.0 / self.qps        # mean of exponential distro

        async def issue(idx):
            prompt = prompts[idx % len(prompts)]
            t0 = time.time()
            _ = mb.generate(prompt, task_type=mb.task)
            return time.time() - t0

        latencies = []
        start = time.time()
        idx = 0
        while time.time() - start < self.duration:
            await asyncio.sleep(random.expovariate(self.qps))
            latencies.append(asyncio.create_task(issue(idx)))
            idx += 1

        finished = [await t for t in latencies]
        return {
            "mean_latency": mean(finished),
            "p95_latency" : sorted(finished)[int(.95*len(finished))],
            "total_reqs"  : len(finished),
            "effective_qps": len(finished)/self.duration,
        }

    # ────────────────────────────────────────────────────────
    def run(self):
        mb = ModelBenchmark(**self.bench_cfg)

        if self.scenario == "single":
            return mb.run(samples=500, batch_size=1)

        if self.scenario == "batch":
            return mb.run(samples=500, batch_size=32)

        if self.scenario == "server":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stats = loop.run_until_complete(self._server_loop(mb))
            mb.close()
            return stats

        raise ValueError("unknown scenario")
