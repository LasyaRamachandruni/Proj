import csv
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import sys
import random

from toy import EStats, MStats, Monitoring

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

    m1.add_monitoring_cycle("v1_1")

    m1.add_monitoring_path("v3_2", "v2_1")

    m1.add_monitoring_cycle("v2_1")

    """
    # todo
    m1.add_monitoring_trail("v1_1", "v1_1") --> cycle
    m1.add_monitoring_trail("v1_1", "v1_2") --> path

    m1.allocate_monitoring_trail[[trail]]
    m1.deallocate
    """

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
