import csv
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import sys
import random


class EStats:   # edge stats

    def __init__(self, start, dest) -> None:
        self.edge   = (start, dest)
        self.start  = start
        self.dest   = dest

        self.num_monitors = 0   # number of monitors on this edge
        self.mstats = []        # list of MStats that cover this edge

        return


    def __str__(self) -> str:
        return f'{self.edge}: {self.num_monitors}\n\t{self.mstats}'



class MStats:   # monitoring stats

    def __init__(self, start, dest, walk: list, weight: float=0.0) -> None:
        # monitoring path/cycle information
        self.start_node = start
        self.end_node   = dest
        self.nodes_list = walk
        if start == dest:
            self.monitor_type = "cycle"
        else:
            self.monitor_type = "path"

        # used for soft failure detection
        self.advertised_weight   = weight
        self.actual_weight       = weight

        return


    def __str__(self) -> str:
        return (f'Monitoring {self.monitor_type}: ({self.start_node}, {self.end_node})'
                f'\n\tadv weight = {self.advertised_weight},\tact weight = {self.actual_weight}'
                f'\n\twalk = {self.nodes_list}')



class Monitoring:   # global data struct

    def __init__(self, broker: nx.Graph, policy: str = "random") -> None:
        # graphs
        self.broker   = copy.deepcopy(broker)   # broker vision of the network
        self.coverage = nx.Graph()              # track coverage graph

        # link selection policy
        self.policy = policy

        # list of monitor nodes
        self.monitors = []  # type = networkx node

        # list tracking the paths, cycles covered by monitoring.
        self.mstats = []    # type = MStats

        # list tracking the number of paths crossing an edge
        self.estats = {}    # type = EStats
        for e1 in self.broker.edges():  # intialize
            e2 = EStats(e1[0], e1[1])
            self.estats[(e1[0], e1[1])] = e2

        # track the number of monitoring paths & cycles.
        self.num_paths  = 0
        self.num_cycles = 0

        return


    def __str__(self) -> str:
        return f'paths: {self.num_paths}, cycles = {self.num_cycles}'


    def add_monitor(self, m) -> None:
        """
        Add node from list of nodes in broker list (gateway nodes) to be a monitor.
        Will not work if node is already a monitor.
        """
        if (m in self.broker.nodes()) and (m not in self.monitors):
            self.monitors.append(m)
        return


    def add_monitors(self, ms: list) -> None:
        """
        Method to add a list of nodes to be monitors
        """
        for m in ms:
            self.add_monitor(m)
        return


    def add_monitoring_cycle(self, m) -> None:
        """
        Create a cycle between with monitor node, m. The monitor, m, must be in
        the list of monitors.
        """
        if m in self.monitors:  # valid monitoring node
            print(m)
            cycle_list = nx.simple_cycles(self.broker)  # list of available cycles in graph
            filter_cycle_list = []
            for c in cycle_list:    # iterate through cycles in graph
                if m in c and c not in filter_cycle_list:   # filter out cycles without monitor
                    filter_cycle_list.append(c)

            # select cycle by policy
            match self.policy:
                case "random":
                    select_c = random.choice(filter_cycle_list)
                case "longest":
                    max_length      = max([len(c) for c in filter_cycle_list])                  # what is the length of the longest cycle
                    max_length_cs   = [(i) for i in filter_cycle_list if len(i) == max_length]  # find all cycles of longest length
                    select_c        = random.choice(max_length_cs)                              # return a random cycle of longest length
                case "least":
                    pc_coverage = [0] * len(filter_cycle_list)  # list of path cycle coverage scores
                    for i in range(len(filter_cycle_list)):     # iterate through all possible cycles
                        c = filter_cycle_list[i] + [filter_cycle_list[i][0]]    # make it a proper cycle
                        for j in range(len(c) - 1):                 # for each edge
                            e = self.estats.get((c[j], c[j+1]))     # get current paths/cycles covering edge
                            if e is None:   # dict return None --> switch ordering
                                e = self.estats.get((c[j+1], c[j]))
                            pc = e.num_monitors                     # add number of paths/cycles to score for cycle
                            pc_coverage[i] += pc
                        pc_coverage[i] /= (len(c)-1)                # normalize based on number of edges (don't prioritize short length paths)

                    min_pc_score = min(pc_coverage)                 # find the minimum score among all cycles
                    least_covered_cycles = [filter_cycle_list[i] for i in range(len(pc_coverage)) if pc_coverage[i] == min_pc_score]    # filter such that only least covered paths remain

                    select_c = random.choice(least_covered_cycles)  # return a random cycle of least coverage
                case _:
                    print("Invalid policy, policy=random used")
                    select_c = random.choice(filter_cycle_list)

            # coverage graph
            nx.add_cycle(self.coverage, select_c)   # add cycle to coverage graph

            # update MStats
            walk = select_c + [select_c[0]]     # make it a proper cycle
            weight = float(nx.path_weight(self.broker, walk, weight="weight"))
            c      = MStats(m, m, walk, weight)
            self.mstats.append(c)

            # update EStats
            for i in range(len(walk) - 1):  # enumerate over walk getting (u, v) pairs
                # get proper estat
                e = self.estats.get((walk[i], walk[i+1]))
                if e is None:   # dict return None --> switch ordering
                    e = self.estats.get((walk[i+1], walk[i]))

                # update values
                e.num_monitors += 1
                e.mstats.append(c)

            # increment num cycles
            self.num_cycles += 1

        return


    def add_monitoring_path(self, s, d) -> None:
        """
        Create a path between source, s, monitor node to destination, d, monitor node.
        Both s, d must be in the list of monitors.
        """
        if (s in self.monitors) and (d in self.monitors):   # valid s,d monitor
            path_list = list(nx.all_simple_paths(self.broker, s, d))

            # select cycle by policy
            match self.policy:
                case "random":
                    select_p = random.choice(list(path_list))
                case "longest":
                    max_length      = max([len(p) for p in path_list])                  # what is the length of the longest path
                    max_length_ps   = [(i) for i in path_list if len(i) == max_length]  # find all paths of longest length
                    select_p        = random.choice(max_length_ps)                      # return a random path of longest length
                case "least":
                    pc_coverage = [0] * len(path_list)          # list of path/cycle coverage scores
                    for i in range(len(path_list)):             # iterate through all possible cycles
                        p = path_list[i]
                        for j in range(len(p) - 1):                 # for each edge
                            e = self.estats.get((p[j], p[j+1]))     # get current paths/cycles covering edge
                            if e is None:   # dict return None --> switch ordering
                                e = self.estats.get((p[j+1], p[j]))
                            pc = e.num_monitors                     # add number of paths/cycles to score for path
                            pc_coverage[i] += pc
                        pc_coverage[i] /= (len(p)-1)                # normalize based on number of edges (don't prioritize short length paths)

                    min_pc_score = min(pc_coverage)                 # find the minimum score among all paths
                    least_covered_paths = [path_list[i] for i in range(len(pc_coverage)) if pc_coverage[i] == min_pc_score] # filter such that only least covered paths remain

                    select_p = random.choice(least_covered_paths)  # return a random cycle of least coverage
                case _:
                    print("Invalid policy, policy=random used")
                    select_p = random.choice(list(path_list))

            # coverage graph
            nx.add_path(self.coverage, select_p)

            # update MStats
            walk    = select_p
            weight  = float(nx.path_weight(self.broker, walk, weight="weight"))
            p       = MStats(s, d, walk, weight)
            self.mstats.append(p)

            # update EStats
            for i in range(len(walk) - 1):  # enumerate over walk getting (u, v) pairs
                # get proper estat
                e = self.estats.get((walk[i], walk[i+1]))
                if e is None:   # dict return None --> switch ordering
                    e = self.estats.get((walk[i+1], walk[i]))
                # update values
                e.num_monitors += 1
                e.mstats.append(p)

            # increment num paths
            self.num_paths += 1
            
        return


    def display_broker(self) -> None:
        """
        Display the broker graph.
        """
        nx.draw(self.broker, with_labels = True)
        plt.show()
        return


    def display_coverage(self) -> None:
        """
        Display the graph of the monitored paths and cycles on the broker graph.
        """
        nx.draw(self.coverage, with_labels = True)
        plt.show()
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
                    print(len(self.broker.edges))
                    localize_sum += len(self.broker.edges)
                else:
                    localize_sum += len(current_localize_list)  # add len to sum
                print(e, "localized edges:", len(current_localize_list), "sum:", localize_sum)

                # revert to normal
                self.broker.edges[e[0], e[1]]["weight"] -= 4
                self.update_monitoring_weights()

            return localize_sum

        else:                               # soft-link failure detected, do not run
            print("")
            return 0


    def localize(self) -> list[EStats]:
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
                    if fm in v.mstats:
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

    def update_monitoring_weights(self) -> None:
        """
        Look at all of the current weights on the graph and update MStats,
        EStats.
        """
        for m in self.mstats:
            m.actual_weight = float(nx.path_weight(self.broker, m.nodes_list, weight="weight"))

        return


    def create_domain(domain_id, nodes, use_star_graph=False):
        """
        Creates a graph for a domain.
        :param domain_id: Unique domain ID (e.g., "D1").
        :param nodes: List of node names for the domain.
        :param use_star_graph: Boolean to indicate whether to use a star graph structure.
        :return: A NetworkX Graph representing the domain.
        """
        G = nx.Graph()
        if use_star_graph:
            # Star graph: Connect all nodes to the first node (center of the star)
            center_node = nodes[0]
            for node in nodes[1:]:
                G.add_edge(center_node, node, weight=1)
        else:
            # Complete graph: Connect every pair of nodes
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    G.add_edge(node1, node2, weight=1)
        return G


if __name__ == "__main__":

    # Domain 1
    D1 = nx.Graph()
    D1.add_edge("v1_1", "v1_2", weight=1)
    D1.add_edge("v1_1", "v1_3", weight=1)
    D1.add_edge("v1_2", "v1_5", weight=1)
    D1.add_edge("v1_3", "v1_4", weight=1)
    D1.add_edge("v1_3", "v1_5", weight=1)
    D1.add_edge("v1_4", "v1_5", weight=1)

    # Domain 2
    D2 = nx.Graph()
    D2.add_edge("v2_1", "v2_2", weight=1)
    D2.add_edge("v2_1", "v2_4", weight=1)
    D2.add_edge("v2_2", "v2_3", weight=1)
    D2.add_edge("v2_2", "v2_5", weight=1)
    D2.add_edge("v2_3", "v2_6", weight=1)
    D2.add_edge("v2_4", "v2_5", weight=1)
    D2.add_edge("v2_5", "v2_6", weight=1)

    # Domain 3
    D3 = nx.Graph()
    D3.add_edge("v3_1", "v3_3", weight=1)
    D3.add_edge("v3_1", "v3_5", weight=1)
    D3.add_edge("v3_2", "v3_4", weight=1)
    D3.add_edge("v3_2", "v3_5", weight=1)
    D3.add_edge("v3_3", "v3_4", weight=1)
    D3.add_edge("v3_3", "v3_5", weight=1)
    D3.add_edge("v3_4", "v3_5", weight=1)

    # master graph - the true graph of the system (no one can see this)
    master_graph = nx.compose(D1, D2)               # union D1, D2, D3
    master_graph = nx.compose(master_graph, D3)
    master_graph.add_edge("v1_1", "v2_1", weight=2) # inter-domain links
    master_graph.add_edge("v1_1", "v3_1", weight=2)
    master_graph.add_edge("v2_2", "v3_1", weight=2)
    master_graph.add_edge("v2_3", "v3_2", weight=2)

    # full mesh GRAPH - what the broker sees
    broker_fullmesh = nx.Graph()
    broker_fullmesh.add_edge("v2_1", "v2_2", weight=1)    # domain 2
    broker_fullmesh.add_edge("v2_1", "v2_3", weight=2)
    broker_fullmesh.add_edge("v2_2", "v2_3", weight=1)
    broker_fullmesh.add_edge("v3_1", "v3_2", weight=2)    # domain 3
    broker_fullmesh.add_edge("v1_1", "v2_1", weight=2)    # inter-domain links
    broker_fullmesh.add_edge("v1_1", "v3_1", weight=2)
    broker_fullmesh.add_edge("v2_2", "v3_1", weight=2)
    broker_fullmesh.add_edge("v2_3", "v3_2", weight=2)

    """
    # simple Graph
    broker_simple = nx.MultiGraph()
    broker_simple.add_edge("D1", "D2")
    broker_simple.add_edge("D1", "D3")
    broker_simple.add_edge("D2", "D3")
    broker_simple.add_edge("D2", "D3")
    nx.draw(broker_simple, with_labels = True)
    plt.show()
    """

    """
    # draw graphs
    nx.draw(D1, with_labels = True)
    plt.show()

    nx.draw(D2, with_labels = True)
    plt.show()

    nx.draw(D3, with_labels = True)
    plt.show()

    nx.draw(master_graph, with_labels = True)
    plt.show()

    nx.draw(broker_fullmesh, with_labels = True)
    plt.show()
    """

    print("\n--------------------\nInitialize Monitoring\n--------------------\n")

    # m1 = Monitoring(broker_fullmesh, "random")      # random        path/cycle selection
    # m1 = Monitoring(broker_fullmesh, "longest")     # longest       path/cycle selection
    m1 = Monitoring(broker_fullmesh, "least")       # least covered path/cycle selection
    print("Policy: ", m1.policy)

    m1.add_monitor("v2_1")
    m1.add_monitor("test")
    m1.add_monitors(["test", "v2_1", "v1_1", "v3_2"])

    m1.print_monitoring_nodes()

    m1.add_monitoring_path("v1_1", "v3_2")
    #m1.display_coverage()

    m1.add_monitoring_cycle("v1_1")
    #m1.display_coverage()

    m1.add_monitoring_path("v3_2", "v2_1")

    m1.add_monitoring_cycle("v2_1")

    #print(m1.get_coverage_percent(True))
    #print(m1.monitors)
    m1.print_num_paths_cycles()
    m1.print_mstats()
    m1.print_estats()

    # --------------------------------------------------------------------------
    print("\n--------------------\nFailure Localization\n--------------------\n")

    # update weight on edge (soft failure simulation)
    m1.broker.edges["v2_1", "v1_1"]["weight"] += 4

    # test updating weights
    m1.update_monitoring_weights()
    m1.print_mstats()

    # localize
    m1.localize()

    # revert to normal
    m1.broker.edges["v2_1", "v1_1"]["weight"] -= 4
    m1.update_monitoring_weights()

    # --------------------------------------------------------------------------
    print("\n--------------------\nGlobal Failure Localization\n--------------------\n")
    score = m1.global_single_link_failure_test()
    print("score: ", score)
