import csv
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import os
import pandas as pd
import pprint
import sys
import random

from optical_monitoring import Edge_Stats, Monitoring_Trail, Optical_Monitoring               # data structs
from optical_monitoring import abstract_domain_star, create_domain_star, draw_graph_color     # functions

if __name__ == "__main__":

    # Domain 1
    D1 = nx.Graph()
    D1.add_edge("d1_v1", "d1_v2", weight=1)
    D1.add_edge("d1_v1", "d1_v3", weight=1)
    D1.add_edge("d1_v2", "d1_v5", weight=1)
    D1.add_edge("d1_v3", "d1_v4", weight=1)
    D1.add_edge("d1_v3", "d1_v5", weight=1)
    D1.add_edge("d1_v4", "d1_v5", weight=1)

    # Domain 2
    D2 = nx.Graph()
    D2.add_edge("d2_v1", "d2_v2", weight=1)
    D2.add_edge("d2_v1", "d2_v4", weight=1)
    D2.add_edge("d2_v2", "d2_v3", weight=1)
    D2.add_edge("d2_v2", "d2_v5", weight=1)
    D2.add_edge("d2_v3", "d2_v6", weight=1)
    D2.add_edge("d2_v4", "d2_v5", weight=1)
    D2.add_edge("d2_v5", "d2_v6", weight=1)

    # Domain 3
    D3 = nx.Graph()
    D3.add_edge("d3_v1", "d3_v3", weight=1)
    D3.add_edge("d3_v1", "d3_v5", weight=1)
    D3.add_edge("d3_v2", "d3_v4", weight=1)
    D3.add_edge("d3_v2", "d3_v5", weight=1)
    D3.add_edge("d3_v3", "d3_v4", weight=1)
    D3.add_edge("d3_v3", "d3_v5", weight=1)
    D3.add_edge("d3_v4", "d3_v5", weight=1)

    # master graph - the true graph of the system (no one can see this)
    physical_graph = nx.compose(D1, D2)               # union D1, D2, D3
    physical_graph = nx.compose(physical_graph, D3)
    physical_graph.add_edge("d1_v1", "d2_v1", weight=2) # inter-domain links
    physical_graph.add_edge("d1_v1", "d3_v1", weight=2)
    physical_graph.add_edge("d2_v2", "d3_v1", weight=2)
    physical_graph.add_edge("d2_v3", "d3_v2", weight=2)

    # full mesh GRAPH - what the broker sees
    broker_fullmesh = nx.Graph()
    broker_fullmesh.add_edge("d2_v1", "d2_v2", weight=1)    # domain 2
    broker_fullmesh.add_edge("d2_v1", "d2_v3", weight=2)
    broker_fullmesh.add_edge("d2_v2", "d2_v3", weight=1)
    broker_fullmesh.add_edge("d3_v1", "d3_v2", weight=2)    # domain 3
    broker_fullmesh.add_edge("d1_v1", "d2_v1", weight=2)    # inter-domain links
    broker_fullmesh.add_edge("d1_v1", "d3_v1", weight=2)
    broker_fullmesh.add_edge("d2_v2", "d3_v1", weight=2)
    broker_fullmesh.add_edge("d2_v3", "d3_v2", weight=2)

    # star mesh graph - what the broker sees
    broker_star = nx.Graph()
    D1_star = abstract_domain_star(D1, ["d1_v1"], "1")
    D2_star = abstract_domain_star(D2, ["d2_v1", "d2_v2", "d2_v3"], "2")
    D3_star = abstract_domain_star(D3, ["d3_v1", "d3_v2"], "3")
    broker_star = nx.compose(D1_star, D2_star)          # union D1, D2, D3
    broker_star = nx.compose(broker_star, D3_star)
    broker_star.add_edge("d1_v1", "d2_v1", weight=2)    # inter-domain links
    broker_star.add_edge("d1_v1", "d3_v1", weight=2)
    broker_star.add_edge("d2_v2", "d3_v1", weight=2)
    broker_star.add_edge("d2_v3", "d3_v2", weight=2)

    # draw_graph_color(physical_graph)
    # draw_graph_color(broker_fullmesh)
    # draw_graph_color(broker_star)

    # output to gml
    # nx.write_gml(physical_graph, "top1_phys.gml")
    # nx.write_gml(broker_fullmesh, "top1_full.gml")
    # nx.write_gml(broker_star, "top1_star.gml")

    #print("\n--------------------\nInitialize Monitoring\n--------------------\n")


    # Create a DataFrame
    data = {"random": [], "longest": [], "least monitored": []}

    for f in range(20):     # !!! run each output file (time steps) !!!
        f += 1
        print(f)

        runs = 1            # default 500
        for i in range(runs):  # !!! how many tests to run (how many times to rerun experiment)!!!
            """
            # Full mesh broker graph
            m1 = Optical_Monitoring(broker_fullmesh, "random")      # random        path/cycle selection
            m2 = Optical_Monitoring(broker_fullmesh, "longest")     # longest       path/cycle selection
            m3 = Optical_Monitoring(broker_fullmesh, "least")       # least covered path/cycle selection
            """

            #"""
            # Star broker graph
            m1 = Optical_Monitoring(broker_star, "random", True)    # random        path/cycle selection
            m2 = Optical_Monitoring(broker_star, "longest", True)   # longest       path/cycle selection
            m3 = Optical_Monitoring(broker_star, "least", True)     # least covered path/cycle selection
            dnc = {"1": 1, "2": 3, "3": 2}
            m1.store_domain_node_count(dnc)
            m2.store_domain_node_count(dnc)
            m3.store_domain_node_count(dnc)
            #"""

            """
            # partial coop (star topology)
            m1.store_domain_info(D2, ["d2_v1", "d2_v2", "d2_v3"], "2")
            m2.store_domain_info(D2, ["d2_v1", "d2_v2", "d2_v3"], "2")
            m3.store_domain_info(D2, ["d2_v1", "d2_v2", "d2_v3"], "2")
            """

            # define monitoring nodes
            m1.add_monitoring_nodes(["d1_v1", "d2_v1", "d2_v2", "d2_v3", "d3_v1", "d3_v2"])  # all nodes
            m2.add_monitoring_nodes(["d1_v1", "d2_v1", "d2_v2", "d2_v3", "d3_v1", "d3_v2"])
            m3.add_monitoring_nodes(["d1_v1", "d2_v1", "d2_v2", "d2_v3", "d3_v1", "d3_v2"])

            # create a trail between start and dest nodes (these can be equal -> cycle)
            for j in range(4):  # !!! number changes the amount of monitoring trails !!!
                """
                # random start and end path
                start   = m1.select_random_monitoring_node()
                dest    = m1.select_random_monitoring_node()
                m1.trail_selection(start, dest)
                m2.trail_selection(start, dest)
                m3.trail_selection(start, dest)
                """

                #"""
                # gnpy service file paths
                m1.outputfile_trail_selection(f"hard failure/gnpy path selection data/inputs/toy_1-20_rounds-20_reqs/output_file_{f}.json")
                m2.outputfile_trail_selection(f"hard failure/gnpy path selection data/inputs/toy_1-20_rounds-20_reqs/output_file_{f}.json")
                m3.outputfile_trail_selection(f"hard failure/gnpy path selection data/inputs/toy_1-20_rounds-20_reqs/output_file_{f}.json")
                #"""

                #score1 = m1.global_single_link_failure_test()
                #score2 = m2.global_single_link_failure_test()
                #score3 = m3.global_single_link_failure_test()
                # print(f"scores:\n\tRandom:{score1}\n\tLongest:{score2}\n\tLeast:{score3}")

            """
            # print stats
            m1.print_domain_tables()    # check domain tables
            m2.print_domain_tables()
            m3.print_domain_tables()

            m1.print_monitoring_nodes() # check monitoring nodes
            m2.print_monitoring_nodes()
            m3.print_monitoring_nodes()

            m1.print_num_paths_cycles() # check trail counts
            m2.print_num_paths_cycles()
            m3.print_num_paths_cycles()

            m1.print_mstats()   # check monitoring trails
            m2.print_mstats()
            m3.print_mstats()

            m1.print_estats()   # check edge coverage statistics
            m2.print_estats()
            m3.print_estats()
            """

            score1 = m1.global_single_link_failure_test()
            # print("\n")
            score2 = m2.global_single_link_failure_test()
            # print("\n")
            score3 = m3.global_single_link_failure_test()
            #print(f"scores:\n\tRandom:{score1}\n\tLongest:{score2}\n\tLeast:{score3}")
            data["random"].append(score1)
            data["longest"].append(score2)
            data["least monitored"].append(score3)

            m1, m2, m3 = None, None, None   # reset

    df = pd.DataFrame(data)

    # to excel file
    # df.to_excel(f'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/top1.xlsx', index=False)

    # to csv
    # df.to_csv(f'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/top1.csv', index=False)
