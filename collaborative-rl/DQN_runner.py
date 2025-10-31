from env_creation import GNPyEnv_Gradual
import networkx as nx
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from toy2 import abstract_domain_star

def run_dqn(model):
    try:
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.dqn.dqn import DQNConfig
        from ray.rllib.connectors.env_to_module import FlattenObservations
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
    project_dir = Path(__file__).resolve().parent
    workspace_dir = project_dir.parent

    def _resolve_path(env_value: Optional[str], default: Path, *, must_exist: bool = True) -> Path:
        if env_value:
            candidate = Path(env_value).expanduser()
            if not candidate.is_absolute():
                candidate = (workspace_dir / candidate).resolve()
            if not must_exist or candidate.exists():
                return candidate
            print(f"[run_dqn] Warning: Path '{candidate}' not found. Falling back to '{default}'.")
        return default

    initial_moni_paths = None
    node_count_dic = None
    num_rounds = 20
    max_services_per_round = 20
    max_monitor_trails = 8
    training_time = 10
    start_recording_timestep = (training_time*1000)-(5* num_rounds * max_monitor_trails)


    # toy 2
    top_id = 2
    default_output_dir = workspace_dir / "data" / "toy_2-20_rounds-20_reqs"
    output_dir_path = _resolve_path(os.getenv("OUTPUT_FILES_DIR_top2"), default_output_dir)
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

    logging_dir_default = project_dir / "logging"
    broken_fibers_dir_default = project_dir / "data"
    logging_dir = _resolve_path(os.getenv("LOGGING_FILE_DIR"), logging_dir_default, must_exist=False)
    broken_fibers_dir = _resolve_path(os.getenv("BROKEN_FIBERS_DIR"), broken_fibers_dir_default)
    if not broken_fibers_dir.exists():
        raise FileNotFoundError(f"Expected broken fibers directory at '{broken_fibers_dir}'")
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging_file = logging_dir / f"top{top_id}-{max_monitor_trails}trails.txt"

    def env_creator(env_config):
        return GNPyEnv_Gradual(output_files_dir=str(output_dir_path), rounds=num_rounds, max_services_per_round=max_services_per_round, 
            broker_graph=broker_fullmesh, max_monitoring_trails=max_monitor_trails, start_recording_timestep=start_recording_timestep,
            logging_file=str(logging_file), 
            broken_fibers=broken_fibers, broken_fibers_dir=str(broken_fibers_dir), node_count_dic=node_count_dic)
    register_env("GNPyEnv_Gradual", env_creator)

    config = (
        DQNConfig()
        .environment("GNPyEnv_Gradual")
        .training(replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 60000,
            "alpha": 0.5,
            "beta": 0.5,
            }
        )
        .env_runners(num_env_runners=1, env_to_module_connector=lambda env: FlattenObservations())
        .resources(num_gpus=int(os.getenv("NUM_GPUS")))
        .framework("torch")
        .experimental(_disable_preprocessor_api=True)
    )

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