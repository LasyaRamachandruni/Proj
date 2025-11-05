from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import Env

from toy2 import Optical_Monitoring

try:  # Optional dependency – allow the project to run without TensorFlow/Keras
    import tensorflow as tf  # type: ignore
    from keras.models import load_model  # type: ignore

    tf.config.run_functions_eagerly(True)
    _HAS_KERAS = True
except Exception:  # pragma: no cover - simply flag the absence of Keras/TF
    load_model = None  # type: ignore
    tf = None  # type: ignore
    _HAS_KERAS = False


def _pad_features(matrix: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or trim feature columns to match *target_dim* (mutating a copy)."""

    matrix = np.asarray(matrix, dtype=np.float32)
    rows, cols = matrix.shape
    if cols == target_dim:
        return matrix
    if cols > target_dim:
        return matrix[:, :target_dim]

    pad = np.zeros((rows, target_dim - cols), dtype=np.float32)
    return np.concatenate([matrix, pad], axis=1)


class GNPyEnv_Gradual(Env):
    """Gymnasium-compliant environment for optical network monitoring.

    The observation is a fixed-length float32 vector composed of:
    - Selected monitoring trails flattened into edge-usage vectors.
    - Candidate trails flattened the same way.
    - (Optional) predicted soft-failure probabilities for each candidate.
    - A small meta feature block containing progress counters and reward terms.
    """

    META_FEATURE_COUNT = 10

    def __init__(
        self,
        output_files_dir: str,
        rounds: int,
        max_services_per_round: int,
        broker_graph: nx.Graph,
        max_monitoring_trails: int,
        start_recording_timestep: int,
        logging_file: str,
        broken_fibers: List[str],
        broken_fibers_dir: str,
        initial_monitoring_paths: Optional[List[List[str]]] = None,
        min_prob_threshold: float = 0.25,
        node_count_dic: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        self.output_files_dir = Path(output_files_dir).expanduser().resolve()
        if not self.output_files_dir.exists():
            raise FileNotFoundError(f"Output files directory not found: {self.output_files_dir}")

        self.broken_fibers_dir = Path(broken_fibers_dir).expanduser().resolve()
        if not self.broken_fibers_dir.exists():
            raise FileNotFoundError(f"Broken fibers directory not found: {self.broken_fibers_dir}")

        self.logging_file = Path(logging_file).expanduser()
        if self.logging_file.parent and not self.logging_file.parent.exists():
            self.logging_file.parent.mkdir(parents=True, exist_ok=True)
        with self.logging_file.open("w", encoding="utf-8") as log_fp:
            log_fp.write("New session\n")

        self.max_rounds = int(rounds)
        self.max_services_per_round = max(1, int(max_services_per_round))
        self.max_monitoring_trails = max(1, int(max_monitoring_trails))
        self.start_recording_timestep = int(start_recording_timestep)
        self.min_prob_threshold = float(min_prob_threshold)
        self.node_count_dic = node_count_dic

        self.initial_monitoring_paths = (
            [list(path) for path in initial_monitoring_paths]
            if initial_monitoring_paths
            else [
                ["dA_v2", "dB_v3", "dB_v4", "dB_v2", "dB_v1", "dA_v1", "dC_v1"],
                ["dD_v2", "dC_v2", "dC_v4", "dC_v3", "dA_v2", "dA_v1", "dC_v1"],
                ["dC_v3", "dA_v2", "dA_v1", "dB_v1", "dB_v2", "dB_v4", "dD_v1", "dD_v2", "dC_v2", "dC_v1"],
                ["dA_v1", "dA_v2", "dB_v3", "dB_v4", "dB_v2", "dD_v1", "dD_v2", "dC_v2", "dC_v4", "dC_v3", "dC_v1"],
            ]
        )
        self.persisted_monitoring_trails = [list(path) for path in self.initial_monitoring_paths]

        self.broker_graph = broker_graph.copy()
        self.node_name_to_id: Dict[str, int] = {}
        self.node_id_to_name: Dict[int, str] = {}
        self.edge_name_to_id: Dict[Tuple[str, str], int] = {}
        self.edge_id_to_name: Dict[int, Tuple[str, str]] = {}

        for idx, node in enumerate(self.broker_graph.nodes()):
            node_name = str(node)
            self.node_name_to_id[node_name] = idx
            self.node_id_to_name[idx] = node_name

        for idx, (u, v) in enumerate(self.broker_graph.edges()):
            edge = (str(u), str(v))
            self.edge_name_to_id[edge] = idx
            self.edge_name_to_id[(edge[1], edge[0])] = idx  # undirected convenience
            self.edge_id_to_name[idx] = edge

        self.num_nodes = len(self.node_name_to_id)
        self.num_edges = len(self.edge_id_to_name)

        self.om = None
        self._rebuild_optical_monitor()

        self.lightpaths_dict: Dict[str, List[dict]] = {}
        self.lightpaths: List[List[int]] = []
        self.lightpaths_edge_vector: List[np.ndarray] = []
        self.lightpaths_osnrs: List[np.ndarray] = []
        self.responses: List[dict] = []
        self.last_pred_probs = np.zeros(0, dtype=np.float32)
        self.last_pred_binary = np.zeros(0, dtype=np.float32)

        self.monitored_trails: List[List[str]] = []
        self.monitored_trails_edge_vector: List[np.ndarray] = []

        self.broken_fibers = broken_fibers or []
        if self.broken_fibers:
            self._validate_broken_fibers()

        self.meta_feature_count = self.META_FEATURE_COUNT
        self.obs_dim = (
            self.max_monitoring_trails * self.num_edges
            + self.max_services_per_round * self.num_edges
            + self.max_services_per_round
            + self.meta_feature_count
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.max_services_per_round)

        self.max_steps_per_episode = max(200, self.max_rounds * 2)
        self.lni_target = 0.5
        self.lni_weight = 0.0
        self.switch_penalty = 0.0
        self.reroute_cost_weight = 0.0

        self.timestep = 0
        self.file_num = 0
        self.curr_score = 0.0
        self.last_lni = 0.0
        self.last_switches = 0.0
        self.last_reroute_cost = 0.0

        self._r_detect = 0.0
        self._r_lni = 0.0
        self._r_switch = 0.0
        self._r_reroute = 0.0

        self.model = None
        model_path = os.getenv("MODEL_PATH")
        if model_path and _HAS_KERAS:
            try:
                self.model = load_model(model_path)
            except Exception as exc:  # pragma: no cover - runtime warning only
                print(f"[GNPyEnv] Failed to load model at {model_path}: {exc}")

    # ---------------------------------------------------------------------
    # Environment core helpers
    # ---------------------------------------------------------------------
    def _rebuild_optical_monitor(self) -> None:
        self.om = Optical_Monitoring(self.broker_graph)
        for node_name in self.node_name_to_id:
            self.om.add_monitoring_node(node_name)

    def _validate_broken_fibers(self) -> None:
        available = {path.name for path in self.broken_fibers_dir.iterdir() if path.is_dir()}
        for fiber in self.broken_fibers:
            transformed = fiber.replace("/", " of ")
            if not any(transformed in item for item in available):
                raise ValueError(
                    f"Incorrect fiber name '{transformed}'. Available: {sorted(available)}"
                )

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------
    def _normalize_node_names(self, trail: Sequence[str | int]) -> List[str]:
        names = []
        for node in trail:
            if isinstance(node, str):
                names.append(node)
            elif isinstance(node, (int, np.integer)):
                names.append(self.node_id_to_name[int(node)])
            else:
                raise TypeError(f"Unsupported node type: {type(node)}")
        return names

    def translate_trail(self, trail: Sequence[str | int], translate_type: str) -> List[str | int]:
        match translate_type:
            case "id to name":
                return self._normalize_node_names(trail)
            case "name to id":
                names = self._normalize_node_names(trail)
                return [self.node_name_to_id[name] for name in names]
            case _:
                raise ValueError(f"Unsupported translate_type: {translate_type}")

    def translate_trail_to_edge_ids(self, trail: Sequence[str | int]) -> List[int]:
        names = self._normalize_node_names(trail)
        ids: List[int] = []
        for u, v in zip(names[:-1], names[1:]):
            if (u, v) in self.edge_name_to_id:
                ids.append(self.edge_name_to_id[(u, v)])
            elif (v, u) in self.edge_name_to_id:
                ids.append(self.edge_name_to_id[(v, u)])
            else:
                raise ValueError(f"Edge ({u}, {v}) does not exist in broker graph")
        return ids

    def translate_trail_to_edge_vector(self, trail: Sequence[str | int]) -> np.ndarray:
        names = self._normalize_node_names(trail)
        vector = np.zeros(self.num_edges, dtype=np.float32)
        for u, v in zip(names[:-1], names[1:]):
            if (u, v) in self.edge_name_to_id:
                vector[self.edge_name_to_id[(u, v)]] += 1.0
            elif (v, u) in self.edge_name_to_id:
                vector[self.edge_name_to_id[(v, u)]] += 1.0
            else:
                raise ValueError(f"Edge ({u}, {v}) does not exist in broker graph")
        return vector

    # ------------------------------------------------------------------
    # Episode/state management
    # ------------------------------------------------------------------
    def _select_initial_trails(self) -> List[List[str]]:
        if self.persisted_monitoring_trails:
            return [list(path) for path in self.persisted_monitoring_trails[-self.max_monitoring_trails :]]
        return [list(path) for path in self.initial_monitoring_paths[: self.max_monitoring_trails]]

    def _install_monitoring_trail(self, trail: Sequence[str | int]) -> None:
        names = self._normalize_node_names(trail)
        if names in self.monitored_trails:
            return
        self.monitored_trails.append(names)
        self.monitored_trails_edge_vector.append(self.translate_trail_to_edge_vector(names))
        self.om.add_monitoring_trail(names)

    def _persist_trails(self) -> None:
        dedup: List[List[str]] = []
        seen = set()
        for trail in self.monitored_trails:
            key = tuple(trail)
            if key not in seen:
                dedup.append(list(trail))
                seen.add(key)
        self.persisted_monitoring_trails = dedup[-self.max_monitoring_trails :]

    def _extract_file_index(self, filename: str) -> int:
        start = "output_file_"
        end = ".json"
        if start not in filename:
            return 0
        start_idx = filename.find(start) + len(start)
        end_idx = filename.find(end, start_idx)
        try:
            return int(filename[start_idx:end_idx])
        except (TypeError, ValueError):
            return 0

    def _load_random_snapshot(self) -> None:
        candidates = [path for path in self.broken_fibers_dir.iterdir() if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No sub-directories found under {self.broken_fibers_dir}")
        random_dir = random.choice(candidates)
        files = [path for path in random_dir.iterdir() if path.is_file() and path.suffix == ".json"]
        if not files:
            raise FileNotFoundError(f"No JSON snapshots inside {random_dir}")
        chosen_file = random.choice(files)
        cache_key = f"{random_dir.name}/{chosen_file.name}"

        self.file_num = self._extract_file_index(chosen_file.name)
        if cache_key not in self.lightpaths_dict:
            self.lightpaths_dict[cache_key] = self.get_lightpaths(chosen_file)
        self.responses = self.lightpaths_dict[cache_key]

    def _prepare_candidates(self) -> None:
        self.lightpaths.clear()
        self.lightpaths_edge_vector.clear()
        self.lightpaths_osnrs.clear()

        for response in self.responses:
            path_nodes = response.get("path", [])
            if not path_nodes:
                continue
            path_node_names = self._normalize_node_names(path_nodes)
            path_node_ids = self.translate_trail(path_node_names, "name to id")
            edge_vector = self.translate_trail_to_edge_vector(path_node_names)

            metrics = np.array(
                [
                    float(response.get("OSNR-0.1nm", 0.0)),
                    float(response.get("OSNR-bandwidth", 0.0)),
                    float(response.get("SNR-0.1nm", 0.0)),
                    float(response.get("SNR-bandwidth", 0.0)),
                ],
                dtype=np.float32,
            )

            self.lightpaths.append(path_node_ids)
            self.lightpaths_edge_vector.append(edge_vector)
            self.lightpaths_osnrs.append(metrics)

        self.last_pred_probs = np.zeros(len(self.lightpaths), dtype=np.float32)
        self.last_pred_binary = np.zeros(len(self.lightpaths), dtype=np.float32)

    def _compute_predictions(self) -> None:
        if self.model is None or not self.lightpaths_osnrs:
            self.last_pred_probs = np.zeros(len(self.lightpaths), dtype=np.float32)
            self.last_pred_binary = np.zeros(len(self.lightpaths), dtype=np.float32)
            return

        metrics = np.vstack(self.lightpaths_osnrs).astype(np.float32)
        metrics = _pad_features(metrics.reshape(metrics.shape[0], -1), 16)
        batch = metrics.reshape(metrics.shape[0], 1, 16)

        try:
            preds = self.model.predict(batch, verbose=0)
        except Exception:
            preds = self.model(batch, training=False)

        if tf is not None:
            try:
                preds = tf.convert_to_tensor(preds)
                preds = preds.numpy()
            except (NotImplementedError, AttributeError):
                @tf.function(jit_compile=False)
                def _eager_predict(x):
                    return self.model(x, training=False)

                preds = _eager_predict(tf.convert_to_tensor(batch, dtype=tf.float32))
                preds = preds.numpy()

        preds = np.asarray(preds).reshape(-1).astype(np.float32)
        self.last_pred_probs = np.clip(preds, 0.0, 1.0)
        self.last_pred_binary = (self.last_pred_probs > self.min_prob_threshold).astype(np.float32)

    # ------------------------------------------------------------------
    # Observation & info helpers
    # ------------------------------------------------------------------
    def _compose_observation(self) -> np.ndarray:
        chosen = np.zeros((self.max_monitoring_trails, self.num_edges), dtype=np.float32)
        for idx, vector in enumerate(self.monitored_trails_edge_vector[: self.max_monitoring_trails]):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.shape[0] != self.num_edges:
                arr = np.pad(arr, (0, self.num_edges - arr.shape[0]), mode="constant")
            chosen[idx] = arr
        chosen_flat = chosen.reshape(-1)

        candidates = np.zeros((self.max_services_per_round, self.num_edges), dtype=np.float32)
        for idx, vector in enumerate(self.lightpaths_edge_vector[: self.max_services_per_round]):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.shape[0] != self.num_edges:
                arr = np.pad(arr, (0, self.num_edges - arr.shape[0]), mode="constant")
            candidates[idx] = arr
        candidates_flat = candidates.reshape(-1)

        preds = np.zeros(self.max_services_per_round, dtype=np.float32)
        if len(self.last_pred_probs) > 0:
            length = min(len(self.last_pred_probs), self.max_services_per_round)
            preds[:length] = self.last_pred_probs[:length]

        meta_values = np.array(
            [
                len(self.monitored_trails) / max(1, self.max_monitoring_trails),
                len(self.lightpaths) / max(1, self.max_services_per_round),
                np.tanh(0.001 * float(self.curr_score)),
                np.tanh(0.1 * float(self._r_detect)),
                np.tanh(0.1 * float(self._r_lni)),
                np.tanh(0.1 * float(self._r_switch)),
                np.tanh(0.1 * float(self._r_reroute)),
                self.timestep / max(1, self.max_steps_per_episode),
                1.0 if len(self.lightpaths) == 0 else 0.0,
                1.0 if len(self.monitored_trails) >= self.max_monitoring_trails else 0.0,
            ],
            dtype=np.float32,
        )
        meta = np.zeros(self.meta_feature_count, dtype=np.float32)
        meta[: min(self.meta_feature_count, meta_values.size)] = meta_values[: self.meta_feature_count]

        obs = np.concatenate([chosen_flat, candidates_flat, preds, meta], dtype=np.float32)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _validate_obs(self, obs: np.ndarray, location: str) -> None:
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"{location}: observation is not a numpy array")
        if obs.shape != self.observation_space.shape:
            raise ValueError(f"{location}: observation shape {obs.shape} != {self.observation_space.shape}")
        if obs.dtype != np.float32:
            raise TypeError(f"{location}: observation dtype {obs.dtype} != float32")
        if not np.all(np.isfinite(obs)):
            bad_indices = np.where(~np.isfinite(obs))[0][:8]
            raise ValueError(f"{location}: observation contains non-finite values at indices {bad_indices}")

    def _build_info(self, extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        info: Dict[str, float] = {
            "timestep": float(self.timestep),
            "file_index": float(self.file_num),
            "score": float(self.curr_score),
            "monitored_paths": float(len(self.monitored_trails)),
            "remaining_candidates": float(len(self.lightpaths)),
        }
        if extra:
            info.update(extra)
        return info

    def _append_log(self, payload: Dict[str, float]) -> None:
        serializable = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in payload.items()}
        with self.logging_file.open("a", encoding="utf-8") as log_fp:
            log_fp.write(json.dumps(serializable) + "\n")

    def _get_score(self) -> Tuple[float, int, np.ndarray]:
        if not self.monitored_trails_edge_vector:
            return 0.0, 0, np.zeros(self.num_edges, dtype=np.float32)
        edges_used = np.zeros(self.num_edges, dtype=np.float32)
        for vector in self.monitored_trails_edge_vector:
            arr = np.asarray(vector, dtype=np.float32)
            if arr.shape[0] != self.num_edges:
                arr = np.pad(arr, (0, self.num_edges - arr.shape[0]), mode="constant")
            edges_used += arr
        target_edges = [self.edge_id_to_name[idx] for idx, value in enumerate(edges_used) if value > 0]
        score = self.om.select_link_failure_test(target_edges)
        return float(score), len(target_edges), edges_used

    # ------------------------------------------------------------------
    # Gymnasium Env API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.timestep = 0
        self.curr_score = 0.0
        self._r_detect = 0.0
        self._r_lni = 0.0
        self._r_switch = 0.0
        self._r_reroute = 0.0
        self.last_lni = 0.0
        self.last_switches = 0.0
        self.last_reroute_cost = 0.0

        self._rebuild_optical_monitor()
        self.monitored_trails.clear()
        self.monitored_trails_edge_vector.clear()
        for trail in self._select_initial_trails():
            self._install_monitoring_trail(trail)
        self._persist_trails()

        self._load_random_snapshot()
        self._prepare_candidates()
        self._compute_predictions()

        observation = self._compose_observation()
        self._validate_obs(observation, "reset")
        info = self._build_info()
        return observation, info

    def step(self, action: int):
        self.timestep += 1
        reward = 0.0
        terminated = False
        truncated = False

        if not isinstance(action, (int, np.integer)):
            truncated = True
            reward = -1.0
        else:
            action_idx = int(action)
            if action_idx < 0 or action_idx >= self.max_services_per_round:
                truncated = True
                reward = -1.0
            elif not self.lightpaths:
                terminated = True
            elif action_idx >= len(self.lightpaths):
                reward = -0.1  # discourage invalid choices when fewer candidates remain
            else:
                reward = self._apply_action(action_idx)

        max_trails_reached = len(self.monitored_trails) >= self.max_monitoring_trails
        no_candidates = len(self.lightpaths) == 0
        time_limit_reached = self.timestep >= self.max_steps_per_episode

        if max_trails_reached or no_candidates:
            terminated = True
        if time_limit_reached and not terminated:
            truncated = True

        if terminated and not truncated:
            score, edges_selected, edges_used = self._get_score()
            self.curr_score = score
            self.last_lni = edges_selected / max(1, self.num_edges)
            self.last_switches = max(0.0, self.last_switches)
            self.last_reroute_cost = max(0.0, self.last_reroute_cost)

            if score <= 0.0:
                detection_reward = 0.0
            else:
                detection_reward = float(
                    ((self.num_edges * max(1, edges_selected)) / max(score, 1e-6)) ** 3
                )

            self._r_detect = detection_reward
            self._r_lni = self.lni_weight * (self.last_lni - self.lni_target)
            self._r_reroute = -self.reroute_cost_weight * self.last_reroute_cost

            reward = detection_reward + self._r_lni + self._r_switch + self._r_reroute

            if self.timestep > self.start_recording_timestep:
                self._append_log(
                    {
                        "timestep": self.timestep,
                        "score": self.curr_score,
                        "monitored_paths": len(self.monitored_trails),
                        "edges_selected": edges_selected,
                    }
                )
        elif truncated and reward == 0.0:
            reward = -0.5  # mild penalty for time-limit truncation

        info = self._build_info(
            {
                "time_limit_reached": float(time_limit_reached),
                "reward_total": float(reward),
                "reward_components_detect": float(self._r_detect),
                "reward_components_lni": float(self._r_lni),
                "reward_components_switch": float(self._r_switch),
                "reward_components_reroute": float(self._r_reroute),
            }
        )

        self._compute_predictions()
        observation = self._compose_observation()
        self._validate_obs(observation, "step")

        return observation, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Action logic
    # ------------------------------------------------------------------
    def _apply_action(self, index: int) -> float:
        path_node_ids = self.lightpaths.pop(index)
        path_edge_vector = self.lightpaths_edge_vector.pop(index)
        _ = self.lightpaths_osnrs.pop(index)

        path_node_names = self.translate_trail(path_node_ids, "id to name")
        before_count = len(self.monitored_trails)
        self._install_monitoring_trail(path_node_names)
        after_count = len(self.monitored_trails)

        added_new = after_count > before_count
        if added_new:
            self._persist_trails()

        self._r_switch = -self.switch_penalty * (1.0 if added_new else 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # Data loading utilities
    # ------------------------------------------------------------------
    def get_lightpaths(self, service_file: Path) -> List[dict]:
        service_path = Path(service_file)
        with service_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        responses = data.get("response", [])
        metrics_set = {"SNR-bandwidth", "SNR-0.1nm", "OSNR-bandwidth", "OSNR-0.1nm"}
        results: List[dict] = []

        for response in responses:
            properties = response.get("path-properties")
            if not properties:
                continue

            path_nodes: List[str] = []
            for obj in properties.get("path-route-objects", []):
                hop = obj.get("path-route-object", {}).get("num-unnum-hop")
                if not hop:
                    continue
                node_id = hop.get("node-id")
                if node_id in self.broker_graph.nodes:
                    path_nodes.append(str(node_id))

            if not path_nodes:
                continue

            if self.node_count_dic:
                indices: List[Tuple[int, str]] = []
                for idx in range(len(path_nodes) - 1):
                    left_domain = path_nodes[idx][1]
                    right_domain = path_nodes[idx + 1][1]
                    if left_domain == right_domain:
                        count = self.node_count_dic.get(left_domain)
                        if count and count >= 3:
                            indices.insert(0, (idx + 1, f"d{left_domain}_vC"))
                for idx, node_name in indices:
                    path_nodes.insert(idx, node_name)

            entry = {"path": path_nodes}
            for metric in properties.get("path-metric", []):
                metric_type = metric.get("metric-type")
                if metric_type in metrics_set:
                    entry[metric_type] = float(metric.get("accumulative-value", 0.0))

            results.append(entry)

        return results

