import json
import networkx as nx
import random
from dotenv import load_dotenv
from toy2 import Optical_Monitoring
from gymnasium import Env
from gymnasium.spaces import Discrete, Dict, Box, MultiDiscrete
from ray.rllib.utils.spaces.repeated import Repeated
from typing import List
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
#from f1_score import F1Score
#from tensorflow.config import run_functions_eagerly

class GNPyEnv_Gradual(Env):
	def __init__(self, output_files_dir: str, rounds: int, max_services_per_round: int, broker_graph: nx.classes.graph.Graph, 
		max_monitoring_trails: int, start_recording_timestep: int, logging_file: str, broken_fibers: List[str], 
		broken_fibers_dir: str, initial_monitoring_paths: list = None, min_prob_threshold: float = 0.25, node_count_dic: dict | None = None):
		tf.config.run_functions_eagerly(True)

		#self.model.build(input_shape=(None, None, 16))
		default_model_path = os.path.join(os.path.dirname(__file__), "saved_model.keras")
		model_path_env = os.getenv("MODEL_PATH")
		if model_path_env:
			candidate_path = os.path.abspath(os.path.expanduser(model_path_env))
			if os.path.isfile(candidate_path):
				model_path = candidate_path
			else:
				print(f"[GNPyEnv_Gradual] Warning: MODEL_PATH '{candidate_path}' not found. Falling back to '{default_model_path}'.")
				model_path = default_model_path
		else:
			model_path = default_model_path
		if not os.path.isfile(model_path):
			raise FileNotFoundError(f"MODEL_PATH '{model_path}' does not exist")
		self.model = load_model(model_path)#,
			#custom_objects={"F1Score": F1Score})
		self.model.run_eagerly = True
		input_shape = self.model.input_shape
		print(f"Model input shape: {input_shape}")

		# setting up initial monitoring paths
		if initial_monitoring_paths is not None:
			self.initial_monitoring_paths = initial_monitoring_paths
		else:
			# default initial moni paths
			self.initial_monitoring_paths = [['dA_v2', 'dB_v3', 'dB_v4', 'dB_v2', 'dB_v1', 'dA_v1', 'dC_v1'],
	            ['dD_v2', 'dC_v2', 'dC_v4', 'dC_v3', 'dA_v2', 'dA_v1', 'dC_v1'],
	            ['dC_v3', 'dA_v2', 'dA_v1', 'dB_v1', 'dB_v2', 'dB_v4', 'dD_v1', 'dD_v2', 'dC_v2', 'dC_v1'],
	            ['dA_v1', 'dA_v2', 'dB_v3', 'dB_v4', 'dB_v2', 'dD_v1', 'dD_v2', 'dC_v2', 'dC_v4', 'dC_v3', 'dC_v1']]

		# environment setup
		self.node_count_dic = node_count_dic
		self.start_recording_timestep = start_recording_timestep

		self.max_rounds = rounds

		self.starting_moni_paths = 4
		self.max_monitoring_trails = max_monitoring_trails

		self.max_services_per_round = max_services_per_round
		self.output_files_dir = output_files_dir

		self.broker_graph = broker_graph
		self.lightpaths_dict = {}

		self.timestep = 0
		self.file_num = 0

		self.min_prob_threshold = min_prob_threshold

		#file
		self.logging_file = logging_file
		os.makedirs(os.path.dirname(self.logging_file), exist_ok=True)
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
		print("# of edges:", self.num_edges)

		# setting up Optical Monitoring
		self.monitored_trails = []
		self.monitored_trails_edge_vector = []
		self.persisted_monitor_trails = None
		self.persisted_monitor_trails_edge_vector = None
		self.om = Optical_Monitoring(broker_graph)
		for n in self.node_name_to_id:
			self.om.add_monitoring_node(n)

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
		for fiber_name in broken_fibers:
			found = False
			modded_name = fiber_name.replace('/',' of ')
			for s in sub_dirs_of_broken_fibers:
				if modded_name in s:
					found = True
					break
			if not found:
				raise Exception(f"Incorrect Fiber Name passed in: {modded_name}, expected something from: {sub_dirs_of_broken_fibers}")
		self.broken_fibers_dir = broken_fibers_dir
		self.broken_fibers = broken_fibers

		self.observation_space = Dict({
			# "high" is set to 2 (instead of 1) below to account for cycles that traverse an edge twice
			"chosen moni paths": Repeated(Box(low=0, high=2, shape=(self.num_edges,), dtype=np.int8), max_len=self.max_monitoring_trails),
			"0-1": Repeated(Discrete(n=2, start=0), max_len=self.max_monitoring_trails),
			"new candidate paths": Repeated(Box(low=0, high=2, shape=(self.num_edges,), dtype=np.int8), max_len=self.max_services_per_round)
		})

		self.action_space = Discrete(self.max_services_per_round)

	def translate_trail(self, ls: list, translate_type: str):
		"""
		Given a list of nodes (in id or name form), converts to a list
		of the other form.

        Args:
            ls (list): The trail you wish to convert
            translate_type (str): must match one of the two cases below

        Returns:
            list: The translated list of the other form.
		"""
		retval = None

		match translate_type:
			case "id to name":
				retval = [self.node_id_to_name[i] for i in ls]
			case "name to id":
				retval = [self.node_name_to_id[i] for i in ls]

		return retval

	def translate_trail_to_edge_ids(self, ls: list):
		"""
		Given a list of nodes (in name form), converts to a list
		of edge ids.

        Args:
            ls (list): The trail you wish to convert

        Returns:
            list: The translated list of edge ids
		"""
		retval = []

		for i in range(len(ls) - 1):
			retval.append(self.edge_name_to_id[(ls[i], ls[i+1])] if (ls[i], ls[i+1]) in self.edge_name_to_id else self.edge_name_to_id[(ls[i+1], ls[i])])

		return retval

	def translate_trail_to_edge_vector(self, ls: list):
		"""
		Given a list of nodes (in name form), converts to a vector
		of 0-1s for each edge id.

        Args:
            ls (list): The trail you wish to convert

        Returns:
            list: The vector of 0-1s (size: number of edges in broker graph)
		"""
		retval = [0] * self.num_edges

		for i in range(len(ls) - 1):
			if (ls[i], ls[i+1]) in self.edge_name_to_id:
				retval[self.edge_name_to_id[(ls[i], ls[i+1])]] += 1
			elif (ls[i+1], ls[i]) in self.edge_name_to_id:
				retval[self.edge_name_to_id[(ls[i+1], ls[i])]] += 1
			else:
				raise ValueError("edge does not exist!!!!")

		return retval
	
	def _get_obs(self, reset : bool):
		metrics = []
		edge_vector_ls = []

		if reset:
			self.lightpaths = []
			self.lightpaths_osnrs = []
			self.lightpaths_edge_vector = []

			for d in self.responses:
				# d is a dict. "path" is the key to lead to the path (name format)
				path_as_node_names = d["path"]
				path_as_edge_ids = self.translate_trail_to_edge_ids(path_as_node_names)
				path_as_edge_vector = self.translate_trail_to_edge_vector(path_as_node_names)
				self.lightpaths_edge_vector.append(path_as_edge_vector)
				self.lightpaths.append(path_as_edge_ids)
				self.lightpaths_osnrs.append([
	                d['OSNR-0.1nm'],
	                d['OSNR-bandwidth'],
	                d['SNR-0.1nm'],
	                d['SNR-bandwidth']
	            ])

				# get indexes of chosen moni paths
				if path_as_edge_vector not in edge_vector_ls and path_as_edge_vector in self.monitored_trails_edge_vector:
					edge_vector_ls.append(path_as_edge_vector)
					metrics.append([
					    d['OSNR-0.1nm'],
					    d['OSNR-bandwidth'],
					    d['SNR-0.1nm'],
					    d['SNR-bandwidth']
					])
		else:
			for d in self.responses:
				path_as_node_names = d["path"]
				path_as_edge_vector = self.translate_trail_to_edge_vector(path_as_node_names)
				if path_as_edge_vector not in edge_vector_ls and path_as_edge_vector in self.monitored_trails_edge_vector:
					edge_vector_ls.append(path_as_edge_vector)
					metrics.append([
					    d['OSNR-0.1nm'],
					    d['OSNR-bandwidth'],
					    d['SNR-0.1nm'],
					    d['SNR-bandwidth']
					])

		# changing to 1-d list
		X_test = np.array(metrics)
		X_test = X_test.reshape(-1, 16)
		X_test = np.expand_dims(X_test, axis=1)
		print("X_test: ", X_test)

		Y_pred_probs = self.model.predict_on_batch(X_test.astype(np.float32))
		if not isinstance(Y_pred_probs, np.ndarray):
			Y_pred_probs = np.array(Y_pred_probs)
		print("Y_test: ", Y_pred_probs)
		print("Y_test type: ", type(Y_pred_probs))
		#Y_pred_probs = self.model.predict_on_batch(np.array(X_test).reshape(1,1,16))
		# print(Y_pred_probs, type(Y_pred_probs))
		Y_pred_binary = (Y_pred_probs > self.min_prob_threshold).astype(int)

		return {
			"chosen moni paths": self.monitored_trails_edge_vector,
			"0-1": Y_pred_binary,
			"new candidate paths": self.lightpaths_edge_vector
		}

	def _get_info(self):
		return {"timestep": self.timestep, "output file": self.file_num, "score": self.curr_score}

	def _get_score(self):
		# get edges from all monitoring paths
		edges_used = [0]*self.num_edges
		for v in self.monitored_trails_edge_vector:
			edges_used += v
		target_edges = []
		for i in range(len(edges_used)):
			if edges_used[i] > 0:
				target_edges.append(self.edge_id_to_name[i])

		return self.om.select_link_failure_test(target_edges), len(target_edges)

	def reset(self, seed=None, options=None):
		# picking random file from random subdirectory (broken fiber or regular traffic)
		random_subdir = random.choice(os.listdir(self.broken_fibers_dir))
		files = os.listdir(os.path.join(self.broken_fibers_dir, random_subdir))
		random_filename = random.choice(files)
		start_str = "output_file_"
		end_str = ".json"
		start_index = random_filename.find(start_str) + len(start_str)
		end_index = random_filename.find(end_str, start_index)
		self.file_num = int(random_filename[start_index:end_index])

		if self.file_num not in self.lightpaths_dict:
			self.lightpaths_dict[self.file_num] = self.get_lightpaths(os.path.join(self.broken_fibers_dir, random_subdir) + '/' + random_filename)
		self.responses = self.lightpaths_dict[self.file_num]

		# clearing previous monitoring nodes
		for i in self.monitored_trails:
			self.om.remove_monitoring_trail(i)
		self.monitored_trails.clear()
		self.monitored_trails_edge_vector.clear()

		# determine which monitoring paths should persist into the new episode
		if self.persisted_monitor_trails is None:
			persisted_trails = [list(mp) for mp in self.initial_monitoring_paths]
			persisted_vectors = [self.translate_trail_to_edge_vector(mp) for mp in persisted_trails]
		else:
			persisted_trails = [list(mp) for mp in self.persisted_monitor_trails]
			persisted_vectors = [list(vec) for vec in self.persisted_monitor_trails_edge_vector]

		seen = set()
		for mp, vec in zip(persisted_trails, persisted_vectors):
			if len(self.monitored_trails) >= self.max_monitoring_trails:
				break
			key = tuple(mp)
			if key in seen:
				continue
			seen.add(key)
			self.monitored_trails.append(mp)
			self.om.add_monitoring_trail(mp)
			self.monitored_trails_edge_vector.append(vec)

		self.persisted_monitor_trails = [list(mp) for mp in self.monitored_trails]
		self.persisted_monitor_trails_edge_vector = [list(vec) for vec in self.monitored_trails_edge_vector]

		self.curr_score = None

		info = self._get_info()
		observation = self._get_obs(reset=True)
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
				edge_vector = self.translate_trail_to_edge_vector(chosen_path_as_node_names)
				self.monitored_trails_edge_vector.append(edge_vector)
				self.persisted_monitor_trails = [list(mp) for mp in self.monitored_trails]
				self.persisted_monitor_trails_edge_vector = [list(vec) for vec in self.monitored_trails_edge_vector]
			reward = 0

			self.lightpaths.pop(action)
			self.lightpaths_edge_vector.pop(action)
			self.lightpaths_osnrs.pop(action)
		else:
			# illegal action. Terminating
			reward = -1
			info = {}
			observation = self._get_obs(reset=False)
			terminated = True
			return observation, reward, terminated, True, info

		if len(self.monitored_trails_edge_vector) == self.max_monitoring_trails:
			terminated = True
			self.curr_score, num_edges_selected = self._get_score()
			info = self._get_info()
			reward = ((self.num_edges*num_edges_selected)/self.curr_score)**3

			# writing to logging file
			if self.timestep > self.start_recording_timestep:
				json_string = json.dumps(info)
				with open(self.logging_file, "a") as f:
					f.write(json_string + "\n")
		else:
			# Haven't selected enough trails yet
			info = {}

		observation = self._get_obs(reset=False)
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