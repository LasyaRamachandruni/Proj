import os
import sys
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from env_creation import GNPyEnv_Gradual


def _build_broker_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("dA_v1", "dA_v2", weight=3)
    graph.add_edge("dB_v1", "dB_v2", weight=2)
    graph.add_edge("dB_v1", "dB_v3", weight=2)
    graph.add_edge("dB_v1", "dB_v4", weight=3)
    graph.add_edge("dB_v2", "dB_v3", weight=2)
    graph.add_edge("dB_v2", "dB_v4", weight=1)
    graph.add_edge("dB_v3", "dB_v4", weight=1)
    graph.add_edge("dC_v1", "dC_v2", weight=2)
    graph.add_edge("dC_v1", "dC_v3", weight=2)
    graph.add_edge("dC_v1", "dC_v4", weight=3)
    graph.add_edge("dC_v2", "dC_v3", weight=2)
    graph.add_edge("dC_v2", "dC_v4", weight=1)
    graph.add_edge("dC_v3", "dC_v4", weight=1)
    graph.add_edge("dD_v1", "dD_v2", weight=2)
    graph.add_edge("dA_v1", "dB_v1", weight=2)
    graph.add_edge("dA_v1", "dC_v1", weight=2)
    graph.add_edge("dA_v2", "dB_v3", weight=2)
    graph.add_edge("dA_v2", "dC_v3", weight=2)
    graph.add_edge("dD_v1", "dB_v2", weight=2)
    graph.add_edge("dD_v1", "dB_v4", weight=2)
    graph.add_edge("dD_v2", "dC_v2", weight=2)
    graph.add_edge("dD_v2", "dC_v4", weight=2)
    return graph


def _resolve_path(name: str, *fallbacks: Path) -> Path:
    workspace_dir = CURRENT_DIR.parent
    value = os.getenv(name)
    candidates: list[Path] = []
    if value:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (workspace_dir / p).resolve()
        candidates.append(p)
    candidates.extend(fallbacks)
    for cand in candidates:
        cand = cand.expanduser()
        if not cand.is_absolute():
            cand = (workspace_dir / cand).resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError(f"None of the candidate paths exist for {name}: {candidates}")


def _env_factory(env_config):
    load_dotenv()
    workspace_dir = CURRENT_DIR.parent

    output_dir = _resolve_path(
        "OUTPUT_FILES_DIR_top2",
        workspace_dir / "data" / "toy_2-20_rounds-20_reqs",
        CURRENT_DIR / "data" / "toy_2-20_rounds-20_reqs",
    )
    broken_dir = _resolve_path(
        "BROKEN_FIBERS_DIR",
        CURRENT_DIR / "data",
        workspace_dir / "collaborative-rl" / "data",
        workspace_dir / "data",
    )

    broken_fibers = [
        "fiber (dA_v1 \u2192 dC_v1)_(2/2)",
        "fiber (dD_v2 \u2192 dC_v2)_(2/2)",
        "fiber (dB_v4 \u2192 dD_v1)_(2/2)",
        "fiber (dA_v2 \u2192 dB_v3)_(1/2)",
    ]

    log_dir = Path(os.getenv("LOGGING_FILE_DIR", workspace_dir / "logs")).expanduser()
    if not log_dir.is_absolute():
        log_dir = (workspace_dir / log_dir).resolve()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        log_dir = (workspace_dir / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

    return GNPyEnv_Gradual(
        output_files_dir=str(output_dir),
        rounds=20,
        max_services_per_round=20,
        broker_graph=_build_broker_graph(),
        max_monitoring_trails=8,
        start_recording_timestep=0,
        logging_file=str(log_dir / "run.log"),
        broken_fibers=broken_fibers,
        broken_fibers_dir=str(broken_dir),
    )


def run():
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.dqn import DQNConfig

    env_name = "GNPyEnv-Gradual"
    register_env(env_name, _env_factory)

    cfg = (
        DQNConfig()
        .environment(env_name)
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1, rollout_fragment_length=16)
        .training(
            train_batch_size=64,
            gamma=0.99,
            lr=1e-3,
            dueling=True,
            double_q=True,
        )
    )

    algo = cfg.build()
    for i in range(10):
        result = algo.train()
        stats = result.get("env_runners", {})
        reward_mean = result.get("episode_reward_mean")
        if reward_mean is None:
            reward_mean = stats.get("episode_return_mean")
        len_mean = result.get("episode_len_mean")
        if len_mean is None:
            len_mean = stats.get("episode_len_mean")
        ts_total = result.get("timesteps_total")
        if ts_total is None:
            ts_total = result.get("num_env_steps_sampled_lifetime") or stats.get("num_env_steps_sampled_lifetime")
        print(
            "iter",
            i,
            "reward_mean",
            reward_mean,
            "len_mean",
            len_mean,
            "ts_total",
            ts_total,
        )
    algo.stop()


if __name__ == "__main__":
    run()
