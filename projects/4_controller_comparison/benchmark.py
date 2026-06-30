import logging
import os
import datetime
import yaml

from schema import BenchmarkConfig
from evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark")

def benchmark_flow(cfg: BenchmarkConfig):
    episodes = cfg.run.episodes
    device = cfg.run.device
    output_dir = cfg.run.output_dir
    selected_controllers = cfg.run.controllers
    steps = cfg.run.steps
    render_mode = cfg.run.render
    test_mode = cfg.run.test
    
    logger.info(f"Running Controller Benchmark Flow over {episodes} episodes (steps limit: {steps}).")
    logger.info(f"Device: {device or 'default'} | Output folder: {output_dir}")

    evaluator = Evaluator(benchmark_cfg=cfg, output_dir=output_dir, device_override=device)

    all_controllers = list(cfg.controllers.keys())
    controllers_to_run = [c for c in selected_controllers if c in all_controllers] if selected_controllers else all_controllers

    for name in controllers_to_run:
        evaluator.evaluate_controller(
            name=name,
            episodes=episodes,
            steps_limit=steps,
            render_mode=render_mode,
            test_mode=test_mode,
        )

    logger.info(f"Controller Benchmark Flow Completed Successfully! Results saved to {output_dir}/episodes_results.csv")

def main():
    config_path = os.path.join(os.path.dirname(__file__), "benchmark_config.yaml")
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    benchmark_cfg = BenchmarkConfig(**yaml_data)

    if benchmark_cfg.run.test:
        benchmark_cfg.run.episodes = 1

    if benchmark_cfg.run.output_dir is None:
        # Assuming execution from project root (PYTHONPATH=.)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        benchmark_cfg.run.output_dir = os.path.join(project_root, "outputs", "benchmark", timestamp)
    else:
        benchmark_cfg.run.output_dir = os.path.abspath(benchmark_cfg.run.output_dir)

    benchmark_flow(benchmark_cfg)

if __name__ == "__main__":
    main()
