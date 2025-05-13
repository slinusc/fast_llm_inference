import argparse, json
from benchmark.scenarios import ScenarioRunner

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend")
    ap.add_argument("--task", choices=["summarization","qa","translation"])
    ap.add_argument("--model")
    ap.add_argument("--path")
    ap.add_argument("--scenario", choices=["single","batch","server"])
    ap.add_argument("--qps", type=float, default=1.0)
    ap.add_argument("--duration", type=int, default=60)
    args = ap.parse_args()

    bench_cfg = dict(
        backend=args.backend,
        task=args.task,
        model_name=args.model,
        model_path=args.path,
        verbose=False
    )

    sr = ScenarioRunner(
        bench_cfg, args.scenario, qps=args.qps, duration_s=args.duration
    )

    sr.run()