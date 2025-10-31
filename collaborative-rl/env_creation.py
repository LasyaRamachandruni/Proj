import json
import networkx as nx
import random
from dotenv import load_dotenv
from toy2 import Optical_Monitoring
import math

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import List
import os
import sys
import numpy as np

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
    _HAS_TF = True
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    load_model = None  # type: ignore
    _HAS_TF = False

#from f1_score import F1Score
#from tensorflow.config import run_functions_eagerly


class GNPyEnv_Gradual(Env):
	def __init__(self, output_files_dir: str, rounds: int, max_services_per_round: int, broker_graph: nx.classes.graph.Graph, 
		max_monitoring_trails: int, start_recording_timestep: int, logging_file: str, broken_fibers: List[str], 
		broken_fibers_dir: str, initial_monitoring_paths: list = None, min_prob_threshold: float = 0.25, node_count_dic: dict | None = None):
		self.starting_moni_paths = 4
		self.prediction_width = self.starting_moni_paths
		self.model = None

		if _HAS_TF:
			tf.config.run_functions_eagerly(True)  # type: ignore[attr-defined]
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
			if os.path.isfile(model_path):
				try:
					self.model = load_model(model_path)
					self.model.run_eagerly = True
					if hasattr(self.model, "output_shape") and self.model.output_shape[-1]:
						self.prediction_width = int(self.model.output_shape[-1])
					print(f"Model input shape: {self.model.input_shape}")
				except Exception as exc:
					print(f"[GNPyEnv_Gradual] Warning: unable to load model '{model_path}': {exc}\n"
					      "Continuing without TensorFlow predictions.")
					self.model = None
			else:
				print(f"[GNPyEnv_Gradual] Info: MODEL_PATH '{model_path}' not found."
				      " Continuing without TensorFlow predictions.")
		else:
			print("[GNPyEnv_Gradual] Info: TensorFlow/Keras not available. Predictions disabled.")

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

		self.n_actions = max(1, min(self.max_services_per_round, self.max_services_per_round))
		self.action_space = gym.spaces.Discrete(self.n_actions)

		self.obs_dim = max(128, self.max_monitoring_trails + self.max_services_per_round + self.num_edges)
		self.observation_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(self.obs_dim,),
			dtype=np.float32,
		)

		self._latest_preds = np.zeros((1, self.prediction_width), dtype=np.float32)
		self.last_reward = 0.0
		self.last_switches = 0.0
		self.last_reroute_cost = 0.0
		self.last_lni = 0.0

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
			u, v = ls[i], ls[i + 1]
			if isinstance(u, str):
				edge = (u, v)
				rev_edge = (v, u)
			else:
				edge = (self.node_id_to_name[u], self.node_id_to_name[v])
				rev_edge = (edge[1], edge[0])
			if edge in self.edge_name_to_id:
				retval.append(self.edge_name_to_id[edge])
			elif rev_edge in self.edge_name_to_id:
				retval.append(self.edge_name_to_id[rev_edge])
			else:
				raise ValueError(f"Edge {edge} not found in broker graph")

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
			u, v = ls[i], ls[i + 1]
			if not isinstance(u, str):
				u = self.node_id_to_name[u]
				v = self.node_id_to_name[v]
			edge = (u, v)
			rev_edge = (v, u)
			if edge in self.edge_name_to_id:
				retval[self.edge_name_to_id[edge]] += 1
			elif rev_edge in self.edge_name_to_id:
				retval[self.edge_name_to_id[rev_edge]] += 1
			else:
				raise ValueError("edge does not exist!!!!")

		return retval

	def _get_obs(self, reset: bool) -> np.ndarray:
		metrics: list[list[float]] = []
		edge_vector_ls: list[list[int]] = []

		if reset:
			self.lightpaths = []
			self.lightpaths_osnrs = []
			self.lightpaths_edge_vector = []

			for response in self.responses:
				path_as_node_names = response["path"]
				path_as_node_ids = self.translate_trail(path_as_node_names, "name to id")
				path_as_edge_vector = self.translate_trail_to_edge_vector(path_as_node_names)
				self.lightpaths_edge_vector.append(path_as_edge_vector)
				self.lightpaths.append(path_as_node_ids)
				self.lightpaths_osnrs.append([
					response['OSNR-0.1nm'],
					response['OSNR-bandwidth'],
					response['SNR-0.1nm'],
					response['SNR-bandwidth']
				])

				if path_as_edge_vector not in edge_vector_ls and path_as_edge_vector in self.monitored_trails_edge_vector:
					edge_vector_ls.append(path_as_edge_vector)
					metrics.append([
						response['OSNR-0.1nm'],
						response['OSNR-bandwidth'],
						response['SNR-0.1nm'],
						response['SNR-bandwidth']
					])
		else:
			for response in self.responses:
				path_as_node_names = response["path"]
				path_as_edge_vector = self.translate_trail_to_edge_vector(path_as_node_names)
				if path_as_edge_vector not in edge_vector_ls and path_as_edge_vector in self.monitored_trails_edge_vector:
					edge_vector_ls.append(path_as_edge_vector)
					metrics.append([
						response['OSNR-0.1nm'],
						response['OSNR-bandwidth'],
						response['SNR-0.1nm'],
						response['SNR-bandwidth']
					])

		flat_metrics = np.array(metrics, dtype=np.float32).flatten()
		if flat_metrics.size == 0:
			flat_metrics = np.zeros(16, dtype=np.float32)
		elif flat_metrics.size % 16 != 0:
			pad_len = 16 - (flat_metrics.size % 16)
			flat_metrics = np.pad(flat_metrics, (0, pad_len))
		X_test = flat_metrics.reshape(-1, 16)
		num_samples = X_test.shape[0]
		X_test = np.expand_dims(X_test, axis=1)
		if os.getenv("GNPY_ENV_DEBUG") == "1":
			print("X_test: ", X_test)

		preds = None
		if self.model is not None:
			try:
				preds_array = self.model.predict_on_batch(X_test.astype(np.float32))
				preds_array = np.array(preds_array)
				preds = preds_array.reshape(preds_array.shape[0], -1)
			except Exception as exc:
				print(f"[GNPyEnv_Gradual] Warning: model inference failed ({exc}). Disabling predictions.")
				self.model = None
				self.prediction_width = self.starting_moni_paths
		if preds is None:
			preds = np.zeros((num_samples, self.prediction_width), dtype=np.float32)
		if os.getenv("GNPY_ENV_DEBUG") == "1":
			print("Y_test: ", preds)
			print("Y_test type: ", type(preds))

		self._latest_preds = preds
		self.last_lni = len(self.monitored_trails) / max(1, self.max_monitoring_trails)
		obs_vec = self._to_obs_vec()
		self._validate_obs(obs_vec, "reset" if reset else "step")
		return obs_vec

	def _to_obs_vec(self) -> np.ndarray:
		monitor_fraction = len(self.monitored_trails) / max(1, self.max_monitoring_trails)
		candidate_fraction = len(self.lightpaths) / max(1, self.max_services_per_round)
		time_fraction = self.timestep / max(1, self.max_rounds)
		base = np.array([
			np.clip(monitor_fraction, 0.0, 1.0),
			np.clip(candidate_fraction, 0.0, 1.0),
			np.clip(time_fraction, 0.0, 1.0),
			math.tanh(self.last_reward),
			math.tanh(self.last_switches),
		], dtype=np.float32)

		tail = np.tanh(self._latest_preds.flatten().astype(np.float32))
		remainder = self.obs_dim - base.size
		if remainder <= 0:
			vec = base[: self.obs_dim]
		else:
			if tail.size < remainder:
				tail = np.concatenate([tail, np.zeros(remainder - tail.size, dtype=np.float32)])
			else:
				tail = tail[:remainder]
			vec = np.concatenate([base, tail], dtype=np.float32)

		np.clip(vec, -1.0, 1.0, out=vec)
		return vec.astype(np.float32)

	def _validate_obs(self, obs: np.ndarray, when: str) -> None:
		if not isinstance(obs, np.ndarray):
			raise ValueError(f"{when}: obs is not ndarray, got {type(obs)}")
		if obs.shape != self.observation_space.shape:
			raise ValueError(f"{when}: obs shape {obs.shape} != {self.observation_space.shape}")
		if obs.dtype != np.float32:
			raise ValueError(f"{when}: obs dtype {obs.dtype} != float32")
		if not np.all(np.isfinite(obs)):
			bad = np.where(~np.isfinite(obs))[0][:10]
			raise ValueError(f"{when}: obs contains non-finite values at indices {bad}")

	def _get_info(self):
		return {"timestep": self.timestep, "output file": self.file_num, "score": self.curr_score}

	def _get_score(self):
		edges_used = np.zeros(self.num_edges, dtype=np.int32)
		for trail in self.monitored_trails:
			try:
				edge_ids = self.translate_trail_to_edge_ids(trail)
			except ValueError:
				continue
			for eid in edge_ids:
				if 0 <= eid < self.num_edges:
					edges_used[eid] += 1

		target_edges = []
		for i, count in enumerate(edges_used):
			if count > 0:
				target_edges.append(self.edge_id_to_name[i])

		return self.om.select_link_failure_test(target_edges), len(target_edges)

	def reset(self, *, seed: int | None = None, options: dict | None = None):
		super().reset(seed=seed)
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

		for trail in self.monitored_trails:
			self.om.remove_monitoring_trail(trail)
		self.monitored_trails.clear()
		self.monitored_trails_edge_vector.clear()

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
		self.curr_round = 0
		self.timestep = 0
		self.last_reward = 0.0
		self.last_switches = 0.0
		self.last_reroute_cost = 0.0
		self.last_lni = len(self.monitored_trails) / max(1, self.max_monitoring_trails)
		self._latest_preds = np.zeros_like(self._latest_preds)

		obs = self._get_obs(reset=True)
		self._validate_obs(obs, "reset")
		return obs, {}

	def step(self, action):
		self.timestep += 1
		self.curr_round += 1
		truncated = False
		reward = 0.0
		new_trail_added = False

		if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.n_actions:
			reward = -1.0
			truncated = True
		else:
			if action < len(self.lightpaths):
				chosen_path = self.lightpaths[action]
				chosen_path_as_node_names = self.translate_trail(chosen_path, "id to name")

				if chosen_path_as_node_names not in self.monitored_trails:
					self.monitored_trails.append(chosen_path_as_node_names)
					self.om.add_monitoring_trail(chosen_path_as_node_names)
					edge_vector = self.translate_trail_to_edge_vector(chosen_path_as_node_names)
					self.monitored_trails_edge_vector.append(edge_vector)
					self.persisted_monitor_trails = [list(mp) for mp in self.monitored_trails]
					self.persisted_monitor_trails_edge_vector = [list(vec) for vec in self.monitored_trails_edge_vector]
					reward = 1.0
					new_trail_added = True

				self.lightpaths.pop(action)
				self.lightpaths_edge_vector.pop(action)
				self.lightpaths_osnrs.pop(action)
			else:
				reward = -0.5

		max_trails_reached = len(self.monitored_trails_edge_vector) >= self.max_monitoring_trails
		max_rounds_reached = self.curr_round >= self.max_rounds
		no_candidates = len(self.lightpaths) == 0
		terminated = max_trails_reached or max_rounds_reached or no_candidates

		if terminated:
			try:
				self.curr_score, _ = self._get_score()
			except Exception as exc:
				print(f"[GNPyEnv_Gradual] Warning: score computation failed ({exc})")
				self.curr_score = None

		self.last_switches = 1.0 if new_trail_added else 0.0
		self.last_reroute_cost = float(len(self.lightpaths))
		self.last_reward = reward

		obs = self._get_obs(reset=False)
		info = {"monitored_paths": len(self.monitored_trails),
			"remaining_candidates": len(self.lightpaths),
			"score": self.curr_score,
		}

		return obs, float(reward), bool(terminated), bool(truncated), info

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