from env_creation import GNPyEnv_Gradual, GNPyEnv_One_Shot
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
import networkx as nx
import os
from dotenv import load_dotenv
from toy2 import abstract_domain_star

# Load the variables from the .env file
load_dotenv()

node_count_dic = None
num_rounds = 20
max_services_per_round = 40
max_monitor_trails = 16
training_time = 400
start_recording_timestep = (training_time*1000)-(5* num_rounds * max_monitor_trails)

'''
# toy 2
top_id = 2
output_dir_path = os.getenv("OUTPUT_FILES_DIR_top2")
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

# toy 1
top_id = 1
output_dir_path = os.getenv("OUTPUT_FILES_DIR_top1")
broker_fullmesh = nx.Graph()
broker_fullmesh.add_edge("d2_v1", "d2_v2", weight=1)    # domain 2
broker_fullmesh.add_edge("d2_v1", "d2_v3", weight=2)
broker_fullmesh.add_edge("d2_v2", "d2_v3", weight=1)
broker_fullmesh.add_edge("d3_v1", "d3_v2", weight=2)    # domain 3
broker_fullmesh.add_edge("d1_v1", "d2_v1", weight=2)    # inter-domain links
broker_fullmesh.add_edge("d1_v1", "d3_v1", weight=2)
broker_fullmesh.add_edge("d2_v2", "d3_v1", weight=2)
broker_fullmesh.add_edge("d2_v3", "d3_v2", weight=2)

# toy 1 star mesh graph - what the broker sees
broker_star = nx.Graph()
top_id = 1
output_dir_path = os.getenv("OUTPUT_FILES_DIR_top1")
D1 = nx.Graph()
D1.add_edge("d1_v1", "d1_v2", weight=1)
D1.add_edge("d1_v1", "d1_v3", weight=1)
D1.add_edge("d1_v2", "d1_v5", weight=1)
D1.add_edge("d1_v3", "d1_v4", weight=1)
D1.add_edge("d1_v3", "d1_v5", weight=1)
D1.add_edge("d1_v4", "d1_v5", weight=1)
D2 = nx.Graph()
D2.add_edge("d2_v1", "d2_v2", weight=1)
D2.add_edge("d2_v1", "d2_v4", weight=1)
D2.add_edge("d2_v2", "d2_v3", weight=1)
D2.add_edge("d2_v2", "d2_v5", weight=1)
D2.add_edge("d2_v3", "d2_v6", weight=1)
D2.add_edge("d2_v4", "d2_v5", weight=1)
D2.add_edge("d2_v5", "d2_v6", weight=1)
D3 = nx.Graph()
D3.add_edge("d3_v1", "d3_v3", weight=1)
D3.add_edge("d3_v1", "d3_v5", weight=1)
D3.add_edge("d3_v2", "d3_v4", weight=1)
D3.add_edge("d3_v2", "d3_v5", weight=1)
D3.add_edge("d3_v3", "d3_v4", weight=1)
D3.add_edge("d3_v3", "d3_v5", weight=1)
D3.add_edge("d3_v4", "d3_v5", weight=1)
D1_star = abstract_domain_star(D1, ["d1_v1"], "1")
D2_star = abstract_domain_star(D2, ["d2_v1", "d2_v2", "d2_v3"], "2")
D3_star = abstract_domain_star(D3, ["d3_v1", "d3_v2"], "3")
broker_star = nx.compose(D1_star, D2_star)          # union D1, D2, D3
broker_star = nx.compose(broker_star, D3_star)
broker_star.add_edge("d1_v1", "d2_v1", weight=2)    # inter-domain links
broker_star.add_edge("d1_v1", "d3_v1", weight=2)
broker_star.add_edge("d2_v2", "d3_v1", weight=2)
broker_star.add_edge("d2_v3", "d3_v2", weight=2)
node_count_dic = {"1": 1, "2": 3, "3": 2}
'''
# toy 3 star
top_id = 3
output_dir_path = os.getenv("OUTPUT_FILES_DIR_top3_40reqs")
D1 = nx.Graph()
D1.add_edge("dA_v1", "dA_v5", weight=1)
D1.add_edge("dA_v1", "dA_v6", weight=1)
D1.add_edge("dA_v1", "dA_v7", weight=1)
D1.add_edge("dA_v2", "dA_v4", weight=1)
D1.add_edge("dA_v2", "dA_v5", weight=1)
D1.add_edge("dA_v2", "dA_v7", weight=1)
D1.add_edge("dA_v2", "dA_v8", weight=1)
D1.add_edge("dA_v3", "dA_v4", weight=1)
D1.add_edge("dA_v3", "dA_v6", weight=1)
D1.add_edge("dA_v3", "dA_v8", weight=1)
D1.add_edge("dA_v4", "dA_v6", weight=1)
D1.add_edge("dA_v4", "dA_v7", weight=1)
D1.add_edge("dA_v5", "dA_v7", weight=1)
D1.add_edge("dA_v6", "dA_v7", weight=1)
D1.add_edge("dA_v7", "dA_v8", weight=1)
# Domain B
D2 = nx.Graph()
D2.add_edge("dB_v1", "dB_v2", weight=1)
D2.add_edge("dB_v1", "dB_v3", weight=1)
D2.add_edge("dB_v1", "dB_v5", weight=1)
D2.add_edge("dB_v2", "dB_v3", weight=1)
D2.add_edge("dB_v3", "dB_v4", weight=1)
D2.add_edge("dB_v3", "dB_v5", weight=1)
D2.add_edge("dB_v4", "dB_v5", weight=1)
# Domain C
D3 = nx.Graph()
D3.add_edge("dC_v1", "dC_v5", weight=1)
D3.add_edge("dC_v1", "dC_v6", weight=1)
D3.add_edge("dC_v2", "dC_v3", weight=1)
D3.add_edge("dC_v2", "dC_v5", weight=1)
D3.add_edge("dC_v2", "dC_v7", weight=1)
D3.add_edge("dC_v3", "dC_v6", weight=1)
D3.add_edge("dC_v3", "dC_v8", weight=1)
D3.add_edge("dC_v4", "dC_v6", weight=1)
D3.add_edge("dC_v4", "dC_v8", weight=1)
D3.add_edge("dC_v7", "dC_v8", weight=1)
# Domain D
D4 = nx.Graph()
D4.add_edge("dD_v1", "dD_v2", weight=1)
D4.add_edge("dD_v1", "dD_v11", weight=1)
D4.add_edge("dD_v2", "dD_v3", weight=1)
D4.add_edge("dD_v3", "dD_v5", weight=1)
D4.add_edge("dD_v4", "dD_v5", weight=1)
D4.add_edge("dD_v4", "dD_v8", weight=1)
D4.add_edge("dD_v4", "dD_v11", weight=1)
D4.add_edge("dD_v5", "dD_v6", weight=1)
D4.add_edge("dD_v6", "dD_v8", weight=1)
D4.add_edge("dD_v7", "dD_v10", weight=1)
D4.add_edge("dD_v7", "dD_v12", weight=1)
D4.add_edge("dD_v8", "dD_v10", weight=1)
D4.add_edge("dD_v9", "dD_v10", weight=1)
D4.add_edge("dD_v9", "dD_v12", weight=1)
D4.add_edge("dD_v11", "dD_v12", weight=1)
# Domain E
D5 = nx.Graph()
D5.add_edge("dE_v1", "dE_v4", weight=1)
D5.add_edge("dE_v1", "dE_v6", weight=1)
D5.add_edge("dE_v2", "dE_v3", weight=1)
D5.add_edge("dE_v2", "dE_v5", weight=1)
D5.add_edge("dE_v2", "dE_v6", weight=1)
D5.add_edge("dE_v3", "dE_v4", weight=1)
D5.add_edge("dE_v3", "dE_v6", weight=1)
D5.add_edge("dE_v4", "dE_v6", weight=1)
D5.add_edge("dE_v5", "dE_v6", weight=1)
# Domain F
D6 = nx.Graph()
D6.add_edge("dF_v1", "dF_v2", weight=1)
D6.add_edge("dF_v1", "dF_v5", weight=1)
D6.add_edge("dF_v1", "dF_v6", weight=1)
D6.add_edge("dF_v2", "dF_v3", weight=1)
D6.add_edge("dF_v2", "dF_v4", weight=1)
D6.add_edge("dF_v2", "dF_v6", weight=1)
D6.add_edge("dF_v3", "dF_v7", weight=1)
D6.add_edge("dF_v4", "dF_v5", weight=1)
D6.add_edge("dF_v4", "dF_v6", weight=1)
D6.add_edge("dF_v6", "dF_v7", weight=1)
broker_fullmesh = nx.Graph()
dA_star = abstract_domain_star(D1, ["dA_v1", "dA_v2", "dA_v3", "dA_v4"], "A")
dB_star = abstract_domain_star(D2, ["dB_v1", "dB_v2", "dB_v3", "dB_v4"], "B")
dC_star = abstract_domain_star(D3, ["dC_v1", "dC_v2", "dC_v3", "dC_v4"], "C")
dD_star = abstract_domain_star(D4, ["dD_v1", "dD_v2", "dD_v3", "dD_v4", "dD_v5", "dD_v6", "dD_v7", "dD_v8", "dD_v9", "dD_v10"], "D")
dE_star = abstract_domain_star(D5, ["dE_v1", "dE_v2", "dE_v3", "dE_v4"], "E")
dF_star = abstract_domain_star(D6, ["dF_v1", "dF_v2", "dF_v3", "dF_v4"], "F")
broker_fullmesh = nx.compose(dA_star, dB_star)          # union star A, B, C, D, E, F
broker_fullmesh = nx.compose(broker_fullmesh, dC_star)
broker_fullmesh = nx.compose(broker_fullmesh, dD_star)
broker_fullmesh = nx.compose(broker_fullmesh, dE_star)
broker_fullmesh = nx.compose(broker_fullmesh, dF_star)
broker_fullmesh.add_edge("dA_v1", "dB_v1", weight=2)    # inter-domain links
broker_fullmesh.add_edge("dA_v4", "dE_v1", weight=2)
broker_fullmesh.add_edge("dB_v2", "dC_v2", weight=2)
broker_fullmesh.add_edge("dC_v4", "dF_v2", weight=2)
broker_fullmesh.add_edge("dF_v4", "dE_v3", weight=2)
broker_fullmesh.add_edge("dD_v1", "dB_v4", weight=2)
broker_fullmesh.add_edge("dD_v2", "dA_v2", weight=2)
broker_fullmesh.add_edge("dD_v3", "dC_v1", weight=2)
broker_fullmesh.add_edge("dD_v4", "dA_v3", weight=2)
broker_fullmesh.add_edge("dD_v5", "dB_v3", weight=2)
broker_fullmesh.add_edge("dD_v6", "dC_v3", weight=2)
broker_fullmesh.add_edge("dD_v7", "dE_v4", weight=2)
broker_fullmesh.add_edge("dD_v8", "dF_v3", weight=2)
broker_fullmesh.add_edge("dD_v9", "dE_v2", weight=2)
broker_fullmesh.add_edge("dD_v10", "dF_v1", weight=2)
node_count_dic = {"A": 4, "B": 4, "C": 4, "D": 10, "E": 4, "F": 4}


def env_creator(env_config):
    return GNPyEnv_Gradual(output_dir_path, num_rounds, max_services_per_round, 
        broker_fullmesh, max_monitor_trails, start_recording_timestep,
		"/srv/data1/home/soham/optical-projects/rl-model/top"+ str(top_id) +"-" + str(max_monitor_trails) + "trails.txt",
        node_count_dic)
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
    .env_runners(num_env_runners=1)
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=1)
)

config.exploration_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.0,
            "epsilon_timesteps": training_time*0.95,
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