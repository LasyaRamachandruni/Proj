from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import networkx as nx
from dotenv import load_dotenv

try:
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.tune.registry import register_env
except ImportError as exc:  # pragma: no cover - fail fast with guidance
    raise SystemExit(
        "Ray/RLlib is missing. Install with `pip install \"ray[rllib]==2.51.0\"` before running."
    ) from exc


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from env_creation import GNPyEnv_Gradual


DEFAULT_BROKEN_FIBERS = [
    "fiber (dA_v1 \u2192 dC_v1)_(2 of 2)",
    "fiber (dD_v2 \u2192 dC_v2)_(2 of 2)",
    "fiber (dB_v4 \u2192 dD_v1)_(2 of 2)",
    "fiber (dA_v2 \u2192 dB_v3)_(1 of 2)",
]


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _choose_path(env_value: str | None, candidates: list[Path]) -> Path:
    if env_value:
        candidate = _resolve(env_value)
        if candidate.exists():
            return candidate
    for path in candidates:
        resolved = _resolve(path)
        if resolved.exists():
            return resolved
    return _resolve(candidates[-1])


def _load_environment_defaults() -> Dict[str, Path]:
    dotenv_candidates = [CURRENT_DIR / ".env", CURRENT_DIR.parent / ".env"]
    for candidate in dotenv_candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)

    data_candidates = [
        CURRENT_DIR / "data" / "toy_2-20_rounds-20_reqs",
        CURRENT_DIR.parent / "data" / "toy_2-20_rounds-20_reqs",
    ]
    data_dir = _choose_path(os.getenv("OUTPUT_FILES_DIR_top2"), data_candidates)

    broken_candidates = [CURRENT_DIR / "data", CURRENT_DIR.parent / "data"]
    broken_dir = _choose_path(os.getenv("BROKEN_FIBERS_DIR"), broken_candidates)
    log_root = _resolve(os.getenv("LOGGING_FILE_DIR", CURRENT_DIR / "logs"))

    try:
        log_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = CURRENT_DIR / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        log_root = fallback.resolve()

    return {
        "data_dir": data_dir,
        "broken_dir": broken_dir,
        "log_root": log_root,
    }


def _build_broker_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge("dA_v1", "dA_v2", weight=3)
    g.add_edge("dB_v1", "dB_v2", weight=2)
    g.add_edge("dB_v1", "dB_v3", weight=2)
    g.add_edge("dB_v1", "dB_v4", weight=3)
    g.add_edge("dB_v2", "dB_v3", weight=2)
    g.add_edge("dB_v2", "dB_v4", weight=1)
    g.add_edge("dB_v3", "dB_v4", weight=1)
    g.add_edge("dC_v1", "dC_v2", weight=2)
    g.add_edge("dC_v1", "dC_v3", weight=2)
    g.add_edge("dC_v1", "dC_v4", weight=3)
    g.add_edge("dC_v2", "dC_v3", weight=2)
    g.add_edge("dC_v2", "dC_v4", weight=1)
    g.add_edge("dC_v3", "dC_v4", weight=1)
    g.add_edge("dD_v1", "dD_v2", weight=2)

    # inter-domain links
    g.add_edge("dA_v1", "dB_v1", weight=2)
    g.add_edge("dA_v1", "dC_v1", weight=2)
    g.add_edge("dA_v2", "dB_v3", weight=2)
    g.add_edge("dA_v2", "dC_v3", weight=2)
    g.add_edge("dD_v1", "dB_v2", weight=2)
    g.add_edge("dD_v1", "dB_v4", weight=2)
    g.add_edge("dD_v2", "dC_v2", weight=2)
    g.add_edge("dD_v2", "dC_v4", weight=2)

    return g


def _env_factory(env_config: Dict | None = None) -> GNPyEnv_Gradual:
    env_config = env_config or {}
    paths = _load_environment_defaults()

    rounds = int(env_config.get("rounds", 20))
    max_services_per_round = int(env_config.get("max_services_per_round", 20))
    max_monitoring_trails = int(env_config.get("max_monitoring_trails", 6))
    start_recording_timestep = int(env_config.get("start_recording_timestep", 0))
    node_count_dic = env_config.get("node_count_dic")

    logging_file = paths["log_root"] / "run.log"

    return GNPyEnv_Gradual(
        output_files_dir=str(paths["data_dir"]),
        rounds=rounds,
        max_services_per_round=max_services_per_round,
        broker_graph=_build_broker_graph(),
        max_monitoring_trails=max_monitoring_trails,
        start_recording_timestep=start_recording_timestep,
        logging_file=str(logging_file),
        broken_fibers=DEFAULT_BROKEN_FIBERS,
        broken_fibers_dir=str(paths["broken_dir"]),
        node_count_dic=node_count_dic,
    )


def _format_metric(value) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def run(iterations: int = 10) -> None:
    env_name = "GNPyEnv-Gradual"
    register_env(env_name, _env_factory)

    config = (
        DQNConfig()
        .environment(env=env_name)
        .framework("torch")
        .resources(num_gpus=int(os.getenv("NUM_GPUS", "0")))
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            rollout_fragment_length=16,
        )
        .training(
            lr=1e-3,
            gamma=0.99,
            train_batch_size=64,
            dueling=True,
            double_q=True,
        )
    )

    algo = config.build()

    try:
        for iteration in range(iterations):
            result = algo.train()
            stats = result.get("env_runners", {}) or {}

            reward_mean = result.get("episode_reward_mean")
            if reward_mean is None:
                reward_mean = stats.get("episode_return_mean")

            length_mean = result.get("episode_len_mean")
            if length_mean is None:
                length_mean = stats.get("episode_len_mean")

            timesteps = (
                result.get("timesteps_total")
                or result.get("num_env_steps_sampled")
                or stats.get("num_env_steps_sampled")
            )

            print(
                f"iter {iteration:02d} | reward_mean={_format_metric(reward_mean)} "
                f"| len_mean={_format_metric(length_mean)} | timesteps={_format_metric(timesteps)}"
            )
    finally:
        algo.stop()


if __name__ == "__main__":
    iterations = int(os.getenv("TRAINING_ITERATIONS", "10"))
    run(iterations=iterations)

