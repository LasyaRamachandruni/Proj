import os
import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dotenv import load_dotenv
import networkx as nx

from env_creation import GNPyEnv_Gradual
from toy2 import abstract_domain_star

def run_dqn(model):
    try:
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.dqn.dqn import DQNConfig
    except ImportError as exc:  # pragma: no cover - only triggered when Ray missing
        raise RuntimeError(
            "Ray RLlib is required to run DQN. Install a compatible version, e.g.\n"
            "  pip install 'ray[rllib]==2.4.0'\n"
            "or update the project dependencies accordingly."
        ) from exc
    print("Running DQN with model:", model)

    # creating model architecture
    num_trails = 4
    num_features = 4
    output_size = 4

    # Load the variables from the .env file
    load_dotenv()
    project_dir = CURRENT_DIR
    workspace_dir = project_dir.parent

    def _resolve_path(env_value: Optional[str], *defaults: Path, must_exist: bool = True) -> Path:
        candidates: list[Path] = []
        tried: list[Path] = []

        if env_value:
            candidate = Path(env_value).expanduser()
            if not candidate.is_absolute():
                candidate = (workspace_dir / candidate).resolve()
            candidates.append(candidate)

        candidates.extend(defaults or [])
        if not candidates:
            raise ValueError("_resolve_path requires at least one default candidate")

        for cand in candidates:
            cand = cand.expanduser()
            if not cand.is_absolute():
                cand = (workspace_dir / cand).resolve()
            else:
                cand = cand.resolve()
            tried.append(cand)
            if not must_exist or cand.exists():
                return cand

        if must_exist:
            tried_str = "\n  ".join(str(p) for p in tried)
            raise FileNotFoundError(
                "Expected path not found. Checked the following locations:\n  " + tried_str
            )

        return tried[-1]

    initial_moni_paths = None
    node_count_dic = None
    num_rounds = 20
    max_services_per_round = 20
    max_monitor_trails = 8
    training_time = 5
    start_recording_timestep = (training_time*1000)-(5* num_rounds * max_monitor_trails)


    # toy 2
    top_id = 2
    output_dir_path = _resolve_path(
        os.getenv("OUTPUT_FILES_DIR_top2"),
        workspace_dir / "data" / "toy_2-20_rounds-20_reqs",
        project_dir / "data" / "toy_2-20_rounds-20_reqs",
    )
    if not output_dir_path.exists():
        raise FileNotFoundError(f"Expected output files directory at '{output_dir_path}'")
    broker_fullmesh = nx.Graph()
    broker_fullmesh.add_edge("dA_v1", "dA_v2", weight=3)  # domain A
    broker_fullmesh.add_edge("dB_v1", "dB_v2", weight=2)  # domain B
    broker_fullmesh.add_edge("dB_v1", "dB_v3", weight=2)
    broker_fullmesh.add_edge("dB_v1", "dB_v4", weight=3)
    broker_fullmesh.add_edge("dB_v2", "dB_v3", weight=2)
    broker_fullmesh.add_edge("dB_v2", "dB_v4", weight=1)
    broker_fullmesh.add_edge("dB_v3", "dB_v4", weight=1)
    broker_fullmesh.add_edge("dC_v1", "dC_v2", weight=2)  # domain C
    broker_fullmesh.add_edge("dC_v1", "dC_v3", weight=2)
    broker_fullmesh.add_edge("dC_v1", "dC_v4", weight=3)
    broker_fullmesh.add_edge("dC_v2", "dC_v3", weight=2)
    broker_fullmesh.add_edge("dC_v2", "dC_v4", weight=1)
    broker_fullmesh.add_edge("dC_v3", "dC_v4", weight=1)
    broker_fullmesh.add_edge("dD_v1", "dD_v2", weight=2)  # domain D
    broker_fullmesh.add_edge("dA_v1", "dB_v1", weight=2)  # inter-domain links
    broker_fullmesh.add_edge("dA_v1", "dC_v1", weight=2)
    broker_fullmesh.add_edge("dA_v2", "dB_v3", weight=2)
    broker_fullmesh.add_edge("dA_v2", "dC_v3", weight=2)
    broker_fullmesh.add_edge("dD_v1", "dB_v2", weight=2)
    broker_fullmesh.add_edge("dD_v1", "dB_v4", weight=2)
    broker_fullmesh.add_edge("dD_v2", "dC_v2", weight=2)
    broker_fullmesh.add_edge("dD_v2", "dC_v4", weight=2)
    '''
    # toy 2 star mesh graph - what the broker sees
    top_id = 2
    output_dir_path = os.getenv("OUTPUT_FILES_DIR_top2")
    D1 = nx.Graph()
    D1.add_edge("dA_v1", "dA_v3", weight=1)
    D1.add_edge("dA_v2", "dA_v4", weight=1)
    D1.add_edge("dA_v3", "dA_v4", weight=1)
    D2 = nx.Graph()
    D2.add_edge("dB_v1", "dB_v5", weight=1)
    D2.add_edge("dB_v2", "dB_v4", weight=1)
    D2.add_edge("dB_v2", "dB_v5", weight=1)
    D2.add_edge("dB_v3", "dB_v4", weight=1)
    D2.add_edge("dB_v3", "dB_v5", weight=1)
    D3 = nx.Graph()
    D3.add_edge("dC_v1", "dC_v5", weight=1)
    D3.add_edge("dC_v2", "dC_v4", weight=1)
    D3.add_edge("dC_v2", "dC_v5", weight=1)
    D3.add_edge("dC_v3", "dC_v4", weight=1)
    D3.add_edge("dC_v3", "dC_v5", weight=1)
    D4 = nx.Graph()
    D4.add_edge("dD_v1", "dD_v3", weight=1)
    D4.add_edge("dD_v2", "dD_v3", weight=1)
    broker_star = nx.Graph()
    dA_star = abstract_domain_star(D1, ["dA_v1", "dA_v2"], "A")
    dB_star = abstract_domain_star(D2, ["dB_v1", "dB_v2", "dB_v3", "dB_v4"], "B")
    dC_star = abstract_domain_star(D3, ["dC_v1", "dC_v2", "dC_v3", "dC_v4"], "C")
    dD_star = abstract_domain_star(D4, ["dD_v1", "dD_v2"], "D")
    broker_star = nx.compose(dA_star, dB_star)          # union dA, dB, dC, dD
    broker_star = nx.compose(broker_star, dC_star)
    broker_star = nx.compose(broker_star, dD_star)
    broker_star.add_edge("dA_v1", "dB_v1", weight=2)  # inter-domain links
    broker_star.add_edge("dA_v1", "dC_v1", weight=2)
    broker_star.add_edge("dA_v2", "dB_v3", weight=2)
    broker_star.add_edge("dA_v2", "dC_v3", weight=2)
    broker_star.add_edge("dD_v1", "dB_v2", weight=2)
    broker_star.add_edge("dD_v1", "dB_v4", weight=2)
    broker_star.add_edge("dD_v2", "dC_v2", weight=2)
    broker_star.add_edge("dD_v2", "dC_v4", weight=2)
    node_count_dic = {"A": 2, "B": 4, "C": 4, "D": 2}
    '''

    broken_fibers = [
            "fiber (dA_v1 \u2192 dC_v1)_(2/2)",
            "fiber (dD_v2 \u2192 dC_v2)_(2/2)",
            "fiber (dB_v4 \u2192 dD_v1)_(2/2)",
            "fiber (dA_v2 \u2192 dB_v3)_(1/2)"
        ]

    logging_dir = _resolve_path(
        os.getenv("LOGGING_FILE_DIR"),
        workspace_dir / "logs",
        project_dir / "logs",
        must_exist=False,
    )
    broken_fiber_candidates = [workspace_dir / "data", project_dir / "data"]
    broken_fibers_dir = _resolve_path(
        os.getenv("BROKEN_FIBERS_DIR"),
        *broken_fiber_candidates,
    )

    def _has_fiber_dirs(path: Path) -> bool:
        try:
            return any(child.is_dir() and child.name.startswith("fiber") for child in path.iterdir())
        except FileNotFoundError:
            return False

    if not _has_fiber_dirs(broken_fibers_dir):
        for candidate in broken_fiber_candidates:
            if _has_fiber_dirs(candidate):
                print(f"[run_dqn] Note: Using broken fiber directory '{candidate}'")
                broken_fibers_dir = candidate
                break
    try:
        logging_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_logging = workspace_dir / "logs"
        fallback_logging.mkdir(parents=True, exist_ok=True)
        print(f"[run_dqn] Warning: unable to create logging directory at '{logging_dir}'. "
              f"Using '{fallback_logging}' instead.")
        logging_dir = fallback_logging
    logging_file = logging_dir / f"top{top_id}-{max_monitor_trails}trails.txt"

    def env_creator(env_config):
        return GNPyEnv_Gradual(output_files_dir=str(output_dir_path), rounds=num_rounds, max_services_per_round=max_services_per_round, 
            broker_graph=broker_fullmesh, max_monitoring_trails=max_monitor_trails, start_recording_timestep=start_recording_timestep,
            logging_file=str(logging_file), 
            broken_fibers=broken_fibers, broken_fibers_dir=str(broken_fibers_dir), node_count_dic=node_count_dic)
    register_env("GNPyEnv_Gradual", env_creator)

    config = DQNConfig()
    config = config.environment("GNPyEnv_Gradual")
    config = config.env_runners(num_env_runners=1)
    config = config.resources(num_gpus=int(os.getenv("NUM_GPUS", "0")))
    config = config.framework("torch")

    config._forward_exploration = {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.0,
                "epsilon_timesteps": training_time*0.9,
            }
    algo = config.build()

    for _ in range(training_time):
        algo.train()
    algo.stop()
    '''
    save_result = algo.save()
    path_to_checkpoint = save_result.checkpoint.path
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )
    '''

if __name__ == "__main__":
    run_dqn(None)