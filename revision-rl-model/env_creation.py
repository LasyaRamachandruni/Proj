import json
import networkx as nx
import random
import numpy as np
import pandas as pd
from toy2 import Optical_Monitoring
from gymnasium import Env
from gymnasium.spaces import Discrete, Dict, Box, MultiDiscrete
from ray.rllib.utils.spaces.repeated import Repeated

class GNPyEnv_Gradual(Env):

	def __init__(self, output_files_dir: str, rounds: int, max_services_per_round: int, broker_graph: nx.classes.graph.Graph, 
		max_monitoring_trails: int, start_recording_timestep: int, logging_file: str, node_count_dic: dict | None = None, 
		broken_fibers: list, broken_fibers_dir: str):
		# environment setup
		self.node_count_dic = node_count_dic
		self.start_recording_timestep = start_recording_timestep
		self.curr_round = 1
		self.max_rounds = rounds
		self.max_services_per_round = max_services_per_round
		self.responses = None
		self.output_files_dir = output_files_dir
		self.broker_graph = broker_graph
		self.lightpaths_dict = {}
		self.curr_score = None
		self.timestep = 0
		self.file_num = 0
		print("# of edges:", self.num_edges)

		#file
		self.logging_file = logging_file
		with open(self.logging_file, "w") as f:
			f.write("New sesh\n")

		# assigning each node and edge to a number
		self.node_name_to_id = {}
		self.node_id_to_name = {}
		self.edge_name_to_id = {}
		self.edge_id_to_name = {}
		self.num_nodes = 0
		self.num_edges = 0
		for i in broker_graph.nodes():
			self.node_name_to_id[i] = self.num_nodes
			self.node_id_to_name[self.num_nodes] = i
			self.num_nodes += 1
		for i in broker_graph.edges:
			self.edge_name_to_id[i] = self.num_edges
			self.edge_id_to_name[self.num_edges] = i
			self.num_edges += 1

		#finding longest path length
		self.max_possible_path_length = 0
		for i in broker_graph.nodes:
			for j in broker_graph.nodes:
				if i != j:
					for path in nx.all_simple_paths(broker_graph, i, j):
						self.max_possible_path_length = max(self.max_possible_path_length, len(path))
		print('longest path length:', self.max_possible_path_length)
		if node_count_dic is not None:
			for i in node_count_dic:
				self.max_possible_path_length += node_count_dic[i] - 1

		# checking if broken fibers are valid
		if not os.path.isdir(broken_fibers_dir):
			raise Exception("Fake path for broken fibers directory!")
		sub_dirs_of_broken_fibers = [x[0] for x in os.walk(broken_fibers_dir)]
		edge_set = self.broker_graph.edges
		for fiber_name, (src, dst) in broken_fibers:
			if fiber_name not in sub_dirs_of_broken_fibers:
				raise Exception("Incorrect Fiber Name passed in")
			if src not in fiber_name or dst not in fiber_name or (src, dst) not in edge_set:
				raise Exception("Incorrect src and/or dst node passed in")
		self.broken_fibers_dir = broken_fibers_dir
		self.broken_fibers = broken_fibers

		self.observation_space = Dict({
			"osnrs": Repeated(Box(low=0.0, high=40.0, dtype=np.float16), max_len=self.max_services_per_round),
			"fiber probabilities": Box(low=0.0, high=1.0, shape=(self.num_edges, ), dtype=np.float16),
			"chosen edges": Box(low=0, high=self.max_monitoring_trails, shape=(self.num_edges, ), dtype=np.int8),
			"available paths (as edges)": Repeated(Repeated(Discrete(self.num_nodes), max_len=self.max_possible_path_length), max_len=self.max_services_per_round)
		})

		self.action_space = Discrete(self.max_services_per_round)

	def translate_trail(self, ls: list, translate_type: str):
		retval = None

		match translate_type:
			case "id to name":
				retval = [self.node_id_to_name[i] for i in ls]
			case "name to id":
				retval = [self.node_name_to_id[i] for i in ls]

		return retval

	def translate_trail_to_edge_ids(self, ls: list):
		retval = []

		for i in range(len(ls) - 1):
			retval.append(self.edge_name_to_id[(ls[i], ls[i+1])] if (ls[i], ls[i+1]) in self.edge_name_to_id else self.edge_name_to_id[(ls[i+1], ls[i])])

		return retval

	def _get_obs(self, action: list | None = None):
		self.lightpaths = []
		self.osnrs = []

		for d in self.responses:
			# d is a dict. "path" is the key to lead to the path (name format)
			path_as_node_names = d["path"]
			path_as_edge_ids = self.translate_trail_to_edge_ids(path_as_node_names)
			self.lightpaths.append(path_as_edge_ids)
			self.osnrs.append(d["OSNR-bandwidth"])

		

		

	def _get_info(self):
		return {"timestep": self.timestep, "output file": self.file_num, "score": self.curr_score}

	def reset(self, seed=None, options=None):
		self.file_num += 1
		if self.file_num > self.max_rounds:
			self.file_num = 1
		# self.file_num = random.randint(1, self.max_rounds)
		if self.file_num not in self.lightpaths_dict:
			self.lightpaths_dict[self.file_num] = self.get_lightpaths(self.output_files_dir + "output_file_" + str(self.file_num) + ".json")
		self.responses = self.lightpaths_dict[self.file_num]

		# clearing previous monitoring nodes
		for i in self.monitored_trails:
			self.om.remove_monitoring_trail(i)
		self.monitored_trails.clear()
		self.curr_round = 1
		self.curr_score = self.num_edges**2

		info = self._get_info()
		observation = self._get_obs()

		return observation, info

	def step(self, action):
		self.timestep += 1
		terminated = False
		reward = None

		if action >= self.max_services_per_round:
			print("ERROR")

		if action < len(self.lightpaths):
			chosen_path = self.lightpaths[action] # always chooses a valid path
			chosen_path_as_node_names = self.translate_trail(chosen_path, "id to name")

			if chosen_path_as_node_names not in self.monitored_trails:
				self.monitored_trails.append(chosen_path_as_node_names)
				self.om.add_monitoring_trail(chosen_path_as_node_names)
			reward = 0

			self.lightpaths.pop(action)
		else:
			# illegal action. Terminating
			reward = -1
			info = {}
			observation = self._get_obs()
			terminated = True
			return observation, reward, terminated, True, info

		if self.curr_round == self.max_monitoring_trails:
			terminated = True
			self.curr_score = self.om.global_single_link_failure_test()
			info = self._get_info()
			reward = ((self.num_edges**2)/self.curr_score)**3

			# writing to file
			if self.timestep > self.start_recording_timestep:
				json_string = json.dumps(info)
				with open(self.logging_file, "a") as f:
					f.write(json_string + "\n")
		else:
			# Haven't selected enough trails yet
			info = {}

		observation = self.lightpaths
		self.curr_round += 1
		return observation, reward, terminated, False, info

	def get_lightpaths(self, service_file):
		node_set = set(self.broker_graph.nodes)
		with open(service_file, 'r') as file:
		    # Load the JSON data into a Python dictionary
		    responses = json.load(file)['response']
		metrics = set(["SNR-bandwidth", "SNR-0.1nm", "OSNR-bandwidth", "OSNR-0.1nm"])
		retval = []

		for i in responses:
			if "path-properties" not in i:
				continue

			path_route_objects = i["path-properties"]["path-route-objects"]
			path_metric = i["path-properties"]["path-metric"]
			curr_path = []
			for j in path_route_objects:
				if "num-unnum-hop" in j["path-route-object"]:
					curr_node = j["path-route-object"]["num-unnum-hop"]["node-id"]
					if curr_node in node_set:
						curr_path.append(curr_node)
			
			if len(curr_path) > 0:
				# change path if given node_count_dic for center nodes
				if self.node_count_dic is not None:
					indices = []            # keep track of indices of missing center nodes
					for i in range(len(curr_path) - 1):     # iterate through path
						nc = self.node_count_dic[curr_path[i][1]]            # look up node count table
						if (curr_path[i][1] == curr_path[i+1][1]) and (nc >= 3):        # nodes in equivalent domain and have center node (star)
							indices.insert(0, (i+1, f"d{curr_path[i][1]}_vC"))  # stack
					for (i, s) in indices:  # insert into path
						curr_path.insert(i, s)

				my_dict = {}
				my_dict["path"] = curr_path
				for m in path_metric:
					if m["metric-type"] in metrics:
						my_dict[m["metric-type"]] = m["accumulative-value"]

				retval.append(my_dict)

		return retval