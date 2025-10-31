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

    # Domain A
    D1 = nx.Graph()
    D1.add_edge("vA_1", "vA_3", weight=1)
    D1.add_edge("vA_2", "vA_4", weight=1)
    D1.add_edge("vA_3", "vA_4", weight=1)

    # Domain B
    D2 = nx.Graph()
    D2.add_edge("vB_1", "vB_5", weight=1)
    D2.add_edge("vB_2", "vB_4", weight=1)
    D2.add_edge("vB_2", "vB_5", weight=1)
    D2.add_edge("vB_3", "vB_4", weight=1)
    D2.add_edge("vB_3", "vB_5", weight=1)

    # Domain C
    D3 = nx.Graph()
    D3.add_edge("vC_1", "vC_5", weight=1)
    D3.add_edge("vC_2", "vC_4", weight=1)
    D3.add_edge("vC_2", "vC_5", weight=1)
    D3.add_edge("vC_3", "vC_4", weight=1)
    D3.add_edge("vC_3", "vC_5", weight=1)

    # Domain D
    D4 = nx.Graph()
    D4.add_edge("vD_1", "vD_3", weight=1)
    D4.add_edge("vD_2", "vD_3", weight=1)

    # master graph - the true graph of the system (no one can see this)
    master_graph = nx.compose(D1, D2)               # union A, B, C, D
    master_graph = nx.compose(master_graph, D3)
    master_graph = nx.compose(master_graph, D4)
    master_graph.add_edge("vA_1", "vB_1", weight=2) # inter-domain links
    master_graph.add_edge("vA_1", "vC_1", weight=2)
    master_graph.add_edge("vA_2", "vB_3", weight=2)
    master_graph.add_edge("vA_2", "vC_3", weight=2)
    master_graph.add_edge("vD_1", "vB_2", weight=2)
    master_graph.add_edge("vD_1", "vB_4", weight=2)
    master_graph.add_edge("vD_2", "vC_2", weight=2)
    master_graph.add_edge("vD_2", "vC_4", weight=2)


    # full mesh GRAPH - what the broker sees
    broker_fullmesh = nx.Graph()
    broker_fullmesh.add_edge("vA_1", "vA_2", weight=3)  # domain A
    broker_fullmesh.add_edge("vB_1", "vB_2", weight=2)  # domain B
    broker_fullmesh.add_edge("vB_1", "vB_3", weight=2)
    broker_fullmesh.add_edge("vB_1", "vB_4", weight=3)
    broker_fullmesh.add_edge("vB_2", "vB_3", weight=2)
    broker_fullmesh.add_edge("vB_2", "vB_4", weight=1)
    broker_fullmesh.add_edge("vB_3", "vB_4", weight=1)
    broker_fullmesh.add_edge("vC_1", "vC_2", weight=2)  # domain C
    broker_fullmesh.add_edge("vC_1", "vC_3", weight=2)
    broker_fullmesh.add_edge("vC_1", "vC_4", weight=3)
    broker_fullmesh.add_edge("vC_2", "vC_3", weight=2)
    broker_fullmesh.add_edge("vC_2", "vC_4", weight=1)
    broker_fullmesh.add_edge("vC_3", "vC_4", weight=1)
    broker_fullmesh.add_edge("vD_1", "vD_2", weight=2)  # domain D
    broker_fullmesh.add_edge("vA_1", "vB_1", weight=2)  # inter-domain links
    broker_fullmesh.add_edge("vA_1", "vC_1", weight=2)
    broker_fullmesh.add_edge("vA_2", "vB_3", weight=2)
    broker_fullmesh.add_edge("vA_2", "vC_3", weight=2)
    broker_fullmesh.add_edge("vD_1", "vB_2", weight=2)
    broker_fullmesh.add_edge("vD_1", "vB_4", weight=2)
    broker_fullmesh.add_edge("vD_2", "vC_2", weight=2)
    broker_fullmesh.add_edge("vD_2", "vC_4", weight=2)

    """
    # draw graphs
    nx.draw(D1, with_labels = True)
    plt.show()

    nx.draw(D2, with_labels = True)
    plt.show()

    nx.draw(D3, with_labels = True)
    plt.show()

    nx.draw(D4, with_labels = True)
    plt.show()

    nx.draw(master_graph, with_labels = True)
    plt.show()

    nx.draw(broker_fullmesh, with_labels = True)
    plt.show()
    """

    print("\n--------------------\nInitialize Monitoring\n--------------------\n")

    # m2 = Monitoring(broker_fullmesh, "random")      # random        path/cycle selection
    # m2 = Monitoring(broker_fullmesh, "longest")     # longest       path/cycle selection
    m2 = Monitoring(broker_fullmesh, "least")       # least covered path/cycle selection
    print("Policy: ", m2.policy)

    m2.add_monitor("v2_1")
    m2.add_monitor("test")
    m2.add_monitors(["test", "vA_1", "vB_2", "vB_3", "vB_4", "vC_1", "vC_3", "vC_4", "vD_1", "vD_2"])

    m2.print_monitoring_nodes()

    # cycles
    m2.add_monitoring_cycle("vA_1")
    m2.add_monitoring_cycle("vB_4")
    m2.add_monitoring_cycle("vC_4")
    m2.add_monitoring_cycle("vD_2")

    # paths
    m2.add_monitoring_path("vA_1", "vB_2")
    m2.add_monitoring_path("vC_4", "vB_2")
    m2.add_monitoring_path("vC_3", "vB_3")
    m2.add_monitoring_path("vC_1", "vD_1")

    #print(m2.get_coverage_percent(True))
    #print(m2.monitors)
    m2.print_num_paths_cycles()
    m2.print_mstats()
    m2.print_estats()

    # --------------------------------------------------------------------------
    print("\n--------------------\nFailure Localization\n--------------------\n")

    # update weight on edge (soft failure simulation)
    m2.broker.edges["vA_1", "vB_1"]["weight"] += 4

    # test updating weights
    m2.update_monitoring_weights()
    m2.print_mstats()

    # localize
    m2.localize()

    # revert to normal
    m2.broker.edges["vA_1", "vB_1"]["weight"] -= 4
    m2.update_monitoring_weights()

    # --------------------------------------------------------------------------
    print("\n--------------------\nGlobal Failure Localization\n--------------------\n")
    score = m2.global_single_link_failure_test()
    print("score: ", score)
