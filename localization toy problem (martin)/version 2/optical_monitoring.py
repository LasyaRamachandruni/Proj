import csv
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import os
import pprint
import random
import sys

from typing import Any

class Edge_Stats:   # edge stats

    def __init__(self, start, dest) -> None:
        self.edge   = (start, dest)
        self.start  = start
        self.dest   = dest

        self.num_monitors = 0   # number of monitors on this edge
        self.trails = []        # list of MStats that cover this edge

        return


    def __str__(self) -> str:
        return f'{self.edge}: {self.num_monitors}\n\t{self.trails}'


class Monitoring_Trail: # monitoring stats

    def __init__(self, start, dest, walk: list, weight: float=0.0) -> None:
        self.start_node = start
        self.end_node   = dest
        self.nodes_list = walk
        if start == dest:
            self.monitor_type = "cycle"
        else:
            self.monitor_type = "path"

        # used for soft failure detection
        self.advertised_weight  = weight
        self.actual_weight      = weight

        return


    def __str__(self) -> str:
        return (f'Monitoring {self.monitor_type}: ({self.start_node}, {self.end_node})'
                f'\n\tadv weight = {self.advertised_weight},\tact weight = {self.actual_weight}'
                f'\n\twalk = {self.nodes_list}')


class Optical_Monitoring:

    def __init__(self, broker: nx.Graph, trail_selection_policy: str = "random", star: bool = False) -> None:
        # graphs
        self.broker   = copy.deepcopy(broker)   # broker vision of the network
        self.coverage = nx.Graph()              # track coverage graph  (remove?)

        # trail selection policy
        self.trail_selection_policy = trail_selection_policy
        self.candidate_trail_list   = []    # list of trails to select from, this is to speed up subsequent trail selections in the gnpy path scenario

        # list of monitor nodes
        self.monitors = []  # type = networkx node

        # list tracking the monitoring trails
        self.mstats = []        # type = MStats
        self.num_paths  = 0     # track the number of monitoring trails.
        self.num_cycles = 0

        # list tracking the number of paths crossing an edge
        self.estats = {}    # type = EStats
        for e1 in self.broker.edges():  # intialize
            e2 = Edge_Stats(e1[0], e1[1])
            self.estats[(e1[0], e1[1])] = e2

        # topology type
        self.star = star    # true = star topology, false = fullmesh

        # cooperation level
        self.domain_node_count = {} # used to support star topology with gnpy path usage.
        self.domain_tables = {}     # for when cooperation = true, store domain hop counts as a dict of dictionaries

        return


    def __str__(self) -> str:
        return f'paths: {self.num_paths}, cycles = {self.num_cycles}'


    # --- Monitoring Node Functions ---


    def add_monitoring_node(self, m) -> None:
        """
        Add a node that is a part of the broker graph to a list of monitors.
        This node can serve as an endpoint for a monitoring trail.
        :param m: graph node
        :return: None
        """
        if (m in self.broker.nodes()) and (m not in self.monitors):
            self.monitors.append(m)
        return


    def add_monitoring_nodes(self, ms: list) -> None:
        """
        Add a list of nodes that are a part of the broker graph to the list of
        monitors. These nodes can serve as an endpoints for a monitoring trail.
        :param ms: list of graph nodes [v1, v2, v3, ...]
        :return: None
        """
        for m in ms:
            self.add_monitoring_node(m)
        return


    def remove_monitoring_node(self, m) -> None:
        """
        Remove a node from the list of monitors. This node can no longer serve
        as an endpoint for a monitoring trail.
        :param m: graph node
        :return: None
        """
        if (m in self.monitors):
            self.monitors.remove(m)
        return


    def remove_monitoring_nodes(self, ms: list) -> None:
        """
        Remove a list of nodes from the list of monitors. These nodes can no
        longer serve as an endpoints for a monitoring trail.
        :param ms: list of graph nodes [v1, v2, v3, ...]
        :return: None
        """
        for m in ms:
            self.remove_monitoring_node(m)
        return


    def select_random_monitoring_node(self) -> Any:
        """
        Return a random graph node from the list of graph nodes that can serve
        as monitoring endpoints in the broker graph.
        :param: None
        :return: graph node
        """
        return random.choice(self.monitors)


    # --- Monitoring Trail Functions ---


    def add_monitoring_trail(self, t: list) -> None:
        """
        Add a monitoring trail to the broker graph.
        :param t: monitoring trail to be added. list of nodes [v1, v2, v3, ...]
        :return: None.
        """
        # update MStats
        walk    = t.copy()
        weight  = float(nx.path_weight(self.broker, walk, weight="weight"))

        # check if trail goes through center domain node ("dX_vC")
        nodes_with_vC = [(index, node) for index, node in enumerate(walk) if "vC" in node]
        if len(nodes_with_vC) > 0:
            for (index, node) in nodes_with_vC:
                d = node[:2]            # domain
                if d in self.domain_tables: # check if domain info is stored and update distance weights
                    u = walk[index - 1]     # border nodes
                    v = walk[index + 1]
                    w = self.domain_tables[d][(u, v)]   # get distance between border nodes
                    weight = weight - 2 + w             # no longer hop count of 2, now update to proper distance from table

        mt = Monitoring_Trail(walk[0], walk[-1], walk, weight)
        self.mstats.append(mt)

        # update EStats
        for i in range(len(walk) - 1):  # enumerate over walk getting (u, v) pairs
            # get proper estat
            e = self.estats.get((walk[i], walk[i+1]))
            if e is None:   # dict return None --> switch ordering
                e = self.estats.get((walk[i+1], walk[i]))
            # update values
            e.num_monitors += 1
            e.trails.append(mt)

        # increment num cycles or paths
        if walk[0] == walk[-1]:
            self.num_cycles += 1
        else:
            self.num_paths += 1

        return None


    def remove_monitoring_trail(self, t: list) -> None:
        """
        Remove a monitoring trail from the broker graph if it matches the given
        trail.
        :param t: monitoring trail to be removed. list of nodes [v1, v2, v3, ...]
        :return: None.
        """
        for mt in self.mstats:
            #print(mt.nodes_list)
            #print(t)
            #print(t == mt.nodes_list)
            if t == mt.nodes_list:
                # update MStats
                self.mstats.remove(mt)
                # update EStats
                for i in range(len(t) - 1):  # enumerate over walk getting (u, v) pairs
                    # get proper estat
                    e = self.estats.get((t[i], t[i+1]))
                    if e is None:   # dict return None --> switch ordering
                        e = self.estats.get((t[i+1], t[i]))
                    # update values
                    e.num_monitors -= 1
                    e.trails.remove(mt)

                # decrement num cycles or paths
                if t[0] == t[-1]:
                    self.num_cycles -= 1
                else:
                    self.num_paths -= 1

                break   # end loop

        return None


    def update_monitoring_weights(self) -> None:
        """
        Look at all of the current weights on the graph and update MStats,
        EStats.
        """
        for m in self.mstats:
            walk = m.nodes_list
            weight = float(nx.path_weight(self.broker, walk, weight="weight"))
            # check if trail goes through center domain node ("dX_vC")
            nodes_with_vC = [(index, node) for index, node in enumerate(walk) if "vC" in node]
            if len(nodes_with_vC) > 0:
                for (index, node) in nodes_with_vC:
                    d = node[:2]            # domain
                    if d in self.domain_tables: # check if domain info is stored and update distance weights
                        u = walk[index - 1]     # border nodes
                        v = walk[index + 1]
                        w = self.domain_tables[d][(u, v)]   # get distance between border nodes
                        weight = weight - 2 + w             # no longer hop count of 2, now update to proper distance from table

            m.actual_weight = weight

        return None


    def trail_selection(self, s, d) -> None:
        """
        Given a start and end node that are monitors, get all paths between them.
        If the start node and end node are equivalent, get all cycles. Select
        and add a monitoring trail from the list of paths or cycles based on
        heuristic.
        :param s: start node v1
        :param d: end node v2
        :return: None
        """
        if (s not in self.monitors) and (d not in self.monitors):   # valid s,d monitor
            raise Exception(f"trail_testing Error: invalid start or destination node for montoring trail.")
        candidate_trail_list = []
        if (s == d):    # cycle
            cycle_list = nx.simple_cycles(self.broker)  # list of available cycles in graph
            for c in cycle_list:    # iterate through cycles in graph
                if (s in c) and (c not in candidate_trail_list):   # filter out cycles without monitor
                    while (s != c[0]):      # rotate cycle such that s,d is first in the list
                        c.append(c.pop(0))
                    if c[0] != c[-1]:       # add s,d to the end of the list to make it clear it is a cycle
                        c.append(c[0])
                    candidate_trail_list.append(c)
        else:       # path
            candidate_trail_list = list(nx.all_simple_paths(self.broker, s, d))
        #pprint.pprint(candidate_trail_list)
        c1 = self.trail_select_random(candidate_trail_list)
        c2 = self.trail_select_longest(candidate_trail_list)
        c3 = self.trail_select_least(candidate_trail_list)
        #print("rand:\t", c1, "\nlongest:", c2, "\nleast:\t", c3)
        match self.trail_selection_policy:
            case "random":
                self.add_monitoring_trail(c1)
            case "longest":
                self.add_monitoring_trail(c2)
            case "least":
                self.add_monitoring_trail(c3)

        return None


    def outputfile_trail_selection(self, fp) -> None:
        """
        Given a gnpy output file of a topology snapshot, get a list of candidate
        paths (not trails or cycles). From these candidate paths, select based
        on heuristic and add it to the class.
        :param fp: file path to gnpy output file
        :return: None
        """
        dl = self.get_lightpaths(fp)    # list of dictionaries
        if len(self.candidate_trail_list) < 1:  # empty list
            self.candidate_trail_list = []
            for d in dl:                    # iterate through list
                p = d["path"]               # grab path data only
                if self.star:               # star graph, have to add center nodes into path
                    indices = []            # keep track of indices of missing center nodes
                    for i in range(len(p) - 1):     # iterate through path
                        nc = self.domain_node_count[p[i][1]]            # look up node count table
                        if (p[i][1] == p[i+1][1]) and (nc >= 3):        # nodes in equivalent domain and have center node (star)
                            indices.insert(0, (i+1, f"d{p[i][1]}_vC"))  # stack
                    for (i, s) in indices:  # insert into path
                        p.insert(i, s)
                if len(p) > 1:              # filter for actual paths
                    self.candidate_trail_list.append(p)

        #pprint.pprint(candidate_trail_list)
        c1 = self.trail_select_random(self.candidate_trail_list)
        c2 = self.trail_select_longest(self.candidate_trail_list)
        c3 = self.trail_select_least(self.candidate_trail_list)
        #print("rand:\t", c1, "\nlongest:", c2, "\nleast:\t", c3)
        match self.trail_selection_policy:
            case "random":
                self.add_monitoring_trail(c1)
                self.candidate_trail_list.remove(c1)
            case "longest":
                self.add_monitoring_trail(c2)
                self.candidate_trail_list.remove(c2)
            case "least":
                self.add_monitoring_trail(c3)
                self.candidate_trail_list.remove(c3)

        return None


    def get_lightpaths(self, service_file) -> list:
        """
        Function taken from Soham's RL model. Given a gnpy output file of a
        topology snapshot, get a list of candidate paths (not trails or cycles).
        These are stored as a list of dictionaries containing gnpy simulation
        data.
        :param service_file: file path to gnpy output file
        :return retval: list of dictionaries containing path data [{p1}{p2}{p3}]
        """
        node_set = set(self.broker.nodes)
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
                my_dict = {}
                my_dict["path"] = curr_path
                for m in path_metric:
                    if m["metric-type"] in metrics:
                        my_dict[m["metric-type"]] = m["accumulative-value"]
                retval.append(my_dict)

        return retval


    # --- Heuristics ---


    def trail_select_random(self, candidate_trails: list) -> list:
        """
        Random Trail
        Picks a trail from a list at random.
        :param candidate_trails: list of trails [ [v1,v2,v3...] [u1,u2,u3...] ... ]
        :return t: trail [v1,v2,v3...]
        """
        return random.choice(candidate_trails)


    def trail_select_longest(self, candidate_trails: list) -> list:
        """
        Longest Trail
        Picks a trail based on length.
        :param candidate_trails: list of trails [ [v1,v2,v3...] [u1,u2,u3...] ... ]
        :return t: trail [v1,v2,v3...]
        """
        max_length = max([len(ct) for ct in candidate_trails])  # what is the length of the longest trail
        max_length_cts = [(ct) for ct in candidate_trails if len(ct) == max_length] # find all trails of longest length
        return random.choice(max_length_cts)    # return a random trail of longest length


    def trail_select_least(self, candidate_trails: list) -> list:
        """
        Least Monitored Trail
        Picks a trail based on previous monitoring trail coverage.
        :param candidate_trails: list of trails [ [v1,v2,v3...] [u1,u2,u3...] ... ]
        :return t: trail [v1,v2,v3...]
        """
        candidate_trail_monitoring_scores = [0] * len(candidate_trails)     # list of path cycle coverage scores
        for i in range(len(candidate_trails)):  # iterate through all candidate trails
            ct = candidate_trails[i]
            for j in range(len(ct) - 1):    # iterate through all edges in a candidate_trail
                e = self.estats.get((ct[j], ct[j+1]))   # get current paths/cycles covering edge
                if e is None:   # dict return None --> switch ordering
                    e = self.estats.get((ct[j+1], ct[j]))
                candidate_trail_monitoring_scores[i] += e.num_monitors    # sum the num of trails to score for cycle
            candidate_trail_monitoring_scores[i] /= (len(ct)-1)    # normalize based on hop count (don't prioritize short length paths)

        min_ct_score = min(candidate_trail_monitoring_scores) # find the minimum score among all cycles
        least_covered_trails = [candidate_trails[i] for i in range(len(candidate_trail_monitoring_scores)) if candidate_trail_monitoring_scores[i] == min_ct_score]    # filter such that only least covered paths remain

        return random.choice(least_covered_trails)  # return a random cycle of least coverage


    # --- Tables ---


    def store_domain_info(self, domain_topology: nx.Graph, abstracted_nodes: list, domain_id: str = "1") -> None:
        """

        """
        #print(f"\n\n{abstracted_nodes}")
        d = "d" + domain_id
        self.domain_tables[d] = {}
        if len(abstracted_nodes) > 1:
            for i in range(len(abstracted_nodes)-1):
                src = abstracted_nodes[i]
                for j in range(i+1, len(abstracted_nodes)):
                    dst = abstracted_nodes[j]
                    spl = nx.shortest_path_length(domain_topology, source=src, target=dst)
                    self.domain_tables[d][(src, dst)] = spl
                    self.domain_tables[d][(dst, src)] = spl # reverse is also true

            #pprint.pprint(self.domain_tables)       # pretty print
            #pprint.pprint(self.domain_tables[d])
            #nx.draw(domain_topology, with_labels = True)
            #plt.show()

        else:
            raise Exception(f"store_domain_info Error: function requires at a list with at least nodes to store abstracted data.")

        return


    def store_domain_node_count(self, d: dict[str, int]) -> None:
        """
        Take a dictionary with (domain: num nodes) values.
        This is used to support star topology with gnpy path usage.
        :param d: {"A": 1, "B": 3, "C": 2}
        :return: none
        """
        self.domain_node_count = d

        return None


    # --- Localization Commands ---


    def global_single_link_failure_test(self) -> int:
        """
        Iteratively goes through every edge in the graph and cause soft failure.
        Only 1-link is ever failing per iteration. Counts the number of returned
        links returned by localize() funciton in each iteration. This means,
        the lower the number, the better graph monitoring setup.

        Note: Graph should not be experiencing soft failure in any links before
        running this command. It can lead to false results.

        Given m = number of edges in broker graph, G
        Upper bound = m^2   (return all links per edge; worst case)
        Lower bound = m     (return 1 link per edge; optimal)
        """

        self.update_monitoring_weights()    # make sure weights are properly updated

        sys.stdout = open(os.devnull, 'w')      # disable prints
        intial_localize_list = self.localize()    # get localize list
        sys.stdout = sys.__stdout__             # enable prints

        if len(intial_localize_list) == 0:         # no soft link failure, proceed as normal
            localize_sum = 0
            for e in self.broker.edges:
                # update weight on edge (soft failure simulation)
                self.broker.edges[e[0], e[1]]["weight"] += 4
                self.update_monitoring_weights()

                # localize
                sys.stdout = open(os.devnull, 'w')          # disable prints
                current_localize_list = self.localize()     # get localize list
                sys.stdout = sys.__stdout__                 # enable prints
                if len(current_localize_list) == 0:
                    # print(len(self.broker.edges))
                    localize_sum += len(self.broker.edges)
                else:
                    localize_sum += len(current_localize_list)  # add len to sum
                # print(e, "localized edges:", len(current_localize_list), "sum:", localize_sum)

                # revert to normal
                self.broker.edges[e[0], e[1]]["weight"] -= 4
                self.update_monitoring_weights()

            return localize_sum

        else:                               # soft-link failure detected, do not run
            # print("")
            return 0


    def localize(self) -> list[Edge_Stats]:
        """
        Looks at actual vs advertised_weight across monitored links. Determines
        which edges could be the degradated links.

        Returns the list of suspect edges (the intersection of edges across
        degrading monitoring links).
        """
        # list of monitors are detecting soft failure
        failing_monitors = []
        for m in self.mstats:
            if m.actual_weight != m.advertised_weight:  # soft failure detected
                failing_monitors.append(m)

        # run localization if there are failing monitors
        if len(failing_monitors) > 0:
            # list of edges which share all of these collective monitors
            localized_edges = []
            for k in self.estats:       # key = (start, dest) pair
                v = self.estats.get(k)  # value = estats class object
                affected_edge = True
                for fm in failing_monitors: # match failed monitors with monitors that cover edge
                    if fm in v.trails:
                        continue
                    else:
                        affected_edge = False
                        break
                if affected_edge:   # edge contains all affected monitors
                    localized_edges.append(v)

            print(f'--------------------\nLOCALIZED SOFT FAILURE:\n')
            for e in localized_edges:
                print(e.edge)

            return localized_edges

        else:   # no failing monitors
            print("There is no failure in any monitoring links.")
            return []   # return empty list


    # --- Print commands ---


    def display_broker(self) -> None:
        """
        Display the broker graph.
        """
        draw_graph_color(self.broker)
        return


    def display_coverage(self) -> None:
        """
        Display the graph of the monitored paths and cycles on the broker graph.
        """
        draw_graph_color(self.coverage)
        return


    def get_coverage_percent(self, print_text:bool = False) -> float:
        """
        """
        cov_len = len(self.coverage.edges())
        bro_len = len(self.broker.edges())

        if print_text:
            print("Cover Graph\nEdge Count:\t",  cov_len, "\nEdge List:\t", self.coverage.edges(), "\n")
            print("Broker Graph\nEdge Count:\t", bro_len, "\nEdge List:\t", self.broker.edges())

        return cov_len / bro_len


    def print_num_paths_cycles(self) -> None:
        """
        Prints the total number of monitoring paths and cycles.
        """
        print(f'# Paths: {self.num_paths}\n# Cycles: {self.num_cycles}')
        return


    def print_monitoring_nodes(self) -> None:
        """
        """
        print(f"Monitoring Nodes: {self.monitors}")

        return


    def print_mstats(self) -> None:
        """
        Prints all of the monitoring cycles and paths information.
        """
        print("--------------------\nMSTATS:\n")
        for m in self.mstats:
            print(m)
        return


    def print_estats(self) -> None:
        """
        Print all of the edge stats, including how many paths/cycles cover them
        and which path/cycles.
        """
        print("--------------------\nESTATS:\n")
        for k in self.estats:       # key = (start, dest) pair
            v = self.estats.get(k)  # value = estats class object
            print(v)
        return


    def print_domain_tables(self) -> None:
        """
        Print all of the stored domain distance info stores in tables.
        """
        print("--------------------\nDOMAIN TABLES:\n")
        for d in self.domain_tables:
            print(f"Domain {d}")
            for e in self.domain_tables[d]:
                w = self.domain_tables[d][e]
                print(f"\tBorder Nodes: {e}, Distance: {w}")
            print("\n")
        return


# ------------------------------------------------------------------------------


def abstract_domain_star(domain_topology: nx.Graph, abstracted_nodes: list, domain_id: str = "1") -> nx.Graph:
    """
    Creates a star graph abstraction of a given domain topology.
    :param domain_id: Unique domain ID (e.g., "1" or "A").
    :param domain_topology: nx.Graph topology to be abstracted.
    :param abstracted_nodes: list of nodes to be abstracted and kept in the
        provided topology. These will be the edge nodes of the star.
    :return: A NetworkX Graph representing the abstracted domain.
    """
    G = nx.Graph()
    if len(abstracted_nodes) < 1:
        raise Exception(f"abstract_domain_star Error: Invalid number of nodes to be abstracted.")
    elif len(abstracted_nodes) == 1:  # star graph w/ 1 node doesn't need a center node
        G.add_node(abstracted_nodes[0])
        return G
    elif len(abstracted_nodes) == 2:  # star graph w/ 2 nodes doesn't need a center node
        G.add_edge(abstracted_nodes[0], abstracted_nodes[1], weight=1)
        return G
    else:                           # more than 1 node
        center_node = "d" + domain_id + "_vC"
        old_nodes = domain_topology.nodes()
        for n in abstracted_nodes:
            if n in old_nodes:  # check for valid node
                G.add_edge(center_node, n, weight=1)
            else:
                raise Exception(f"abstract_domain_star Error: Node {n}, is not apart of the provided domain.")
        return G


def create_domain_star(domain_id: str = "1", node_count: int = 4) -> nx.Graph:
    """
    Creates a star graph given a domain ID and number of nodes connecting to
    the center node.
    :param domain_id: Unique domain ID (e.g., "1" or "A").
    :param nodes: the number of nodes being abstracted into a star graph.
    :return: A NetworkX Graph representing the abstracted domain.
    """
    G = nx.Graph()
    # Star graph: Connect all nodes to the first node (center of the star)
    center_node = "d" + domain_id + "_vC"
    for i in range(node_count):
        border_node = "d" + domain_id + "_v" + str(i+1)
        G.add_edge(center_node, border_node, weight=1)
    return G


def draw_graph_color(G: nx.Graph) -> None:
    # Extract unique domain identifiers and map them to colors
    domains = [node.split("_")[0][1:] for node in G.nodes()]
    unique_domains = list(set(domains))  # Get unique domain identifiers
    domain_to_index = {domain: i for i, domain in enumerate(unique_domains)}  # Map domains to indices

    # Create colormap and assign colors
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_domains) - 1)
    node_colors = [cmap(norm(domain_to_index[domain])) for domain in domains]

    # Draw the graph
    nx.draw(G, node_color=node_colors, with_labels=True)
    plt.show()

    return

# ------------------------------------------------------------------------------


if __name__ == "__main__":

    print("test")
