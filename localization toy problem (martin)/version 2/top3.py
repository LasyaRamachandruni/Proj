import csv
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import random

from optical_monitoring import Edge_Stats, Monitoring_Trail, Optical_Monitoring               # data structs
from optical_monitoring import abstract_domain_star, create_domain_star, draw_graph_color     # functions



import time

if __name__ == "__main__":

    # Calculate the start time
    time_start = time.time()

    # Domain A
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
    #draw_graph_color(D1)

    # Domain B
    D2 = nx.Graph()
    D2.add_edge("dB_v1", "dB_v2", weight=1)
    D2.add_edge("dB_v1", "dB_v3", weight=1)
    D2.add_edge("dB_v1", "dB_v5", weight=1)
    D2.add_edge("dB_v2", "dB_v3", weight=1)
    D2.add_edge("dB_v3", "dB_v4", weight=1)
    D2.add_edge("dB_v3", "dB_v5", weight=1)
    D2.add_edge("dB_v4", "dB_v5", weight=1)
    #draw_graph_color(D2)

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
    #draw_graph_color(D3)

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
    #draw_graph_color(D4)

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
    #draw_graph_color(D5)

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
    #draw_graph_color(D6)

    # master graph - the true graph of the system (no one can see this)
    physical_graph = nx.compose(D1, D2)                   # union A, B, C, D, E, F
    physical_graph = nx.compose(physical_graph, D3)
    physical_graph = nx.compose(physical_graph, D4)
    physical_graph = nx.compose(physical_graph, D5)
    physical_graph = nx.compose(physical_graph, D6)
    physical_graph.add_edge("dA_v1", "dB_v1", weight=2)   # inter-domain links
    physical_graph.add_edge("dA_v4", "dE_v1", weight=2)
    physical_graph.add_edge("dB_v2", "dC_v2", weight=2)
    physical_graph.add_edge("dC_v4", "dF_v2", weight=2)
    physical_graph.add_edge("dF_v4", "dE_v3", weight=2)
    physical_graph.add_edge("dD_v1", "dB_v4", weight=2)
    physical_graph.add_edge("dD_v2", "dA_v2", weight=2)
    physical_graph.add_edge("dD_v3", "dC_v1", weight=2)
    physical_graph.add_edge("dD_v4", "dA_v3", weight=2)
    physical_graph.add_edge("dD_v5", "dB_v3", weight=2)
    physical_graph.add_edge("dD_v6", "dC_v3", weight=2)
    physical_graph.add_edge("dD_v7", "dE_v4", weight=2)
    physical_graph.add_edge("dD_v8", "dF_v3", weight=2)
    physical_graph.add_edge("dD_v9", "dE_v2", weight=2)
    physical_graph.add_edge("dD_v10", "dF_v1", weight=2)

    # star mesh graph - what the broker sees
    broker_star = nx.Graph()
    dA_star = abstract_domain_star(D1, ["dA_v1", "dA_v2", "dA_v3", "dA_v4"], "A")
    dB_star = abstract_domain_star(D2, ["dB_v1", "dB_v2", "dB_v3", "dB_v4"], "B")
    dC_star = abstract_domain_star(D3, ["dC_v1", "dC_v2", "dC_v3", "dC_v4"], "C")
    dD_star = abstract_domain_star(D4, ["dD_v1", "dD_v2", "dD_v3", "dD_v4", "dD_v5", "dD_v6", "dD_v7", "dD_v8", "dD_v9", "dD_v10"], "D")
    dE_star = abstract_domain_star(D5, ["dE_v1", "dE_v2", "dE_v3", "dE_v4"], "E")
    dF_star = abstract_domain_star(D6, ["dF_v1", "dF_v2", "dF_v3", "dF_v4"], "F")
    broker_star = nx.compose(dA_star, dB_star)          # union star A, B, C, D, E, F
    broker_star = nx.compose(broker_star, dC_star)
    broker_star = nx.compose(broker_star, dD_star)
    broker_star = nx.compose(broker_star, dE_star)
    broker_star = nx.compose(broker_star, dF_star)
    broker_star.add_edge("dA_v1", "dB_v1", weight=2)    # inter-domain links
    broker_star.add_edge("dA_v4", "dE_v1", weight=2)
    broker_star.add_edge("dB_v2", "dC_v2", weight=2)
    broker_star.add_edge("dC_v4", "dF_v2", weight=2)
    broker_star.add_edge("dF_v4", "dE_v3", weight=2)
    broker_star.add_edge("dD_v1", "dB_v4", weight=2)
    broker_star.add_edge("dD_v2", "dA_v2", weight=2)
    broker_star.add_edge("dD_v3", "dC_v1", weight=2)
    broker_star.add_edge("dD_v4", "dA_v3", weight=2)
    broker_star.add_edge("dD_v5", "dB_v3", weight=2)
    broker_star.add_edge("dD_v6", "dC_v3", weight=2)
    broker_star.add_edge("dD_v7", "dE_v4", weight=2)
    broker_star.add_edge("dD_v8", "dF_v3", weight=2)
    broker_star.add_edge("dD_v9", "dE_v2", weight=2)
    broker_star.add_edge("dD_v10", "dF_v1", weight=2)

    # draw_graph_color(physical_graph)
    # draw_graph_color(broker_star)

    # output to gml
    # nx.write_gml(physical_graph, "top3_phys.gml")
    # nx.write_gml(broker_star, "top3_star.gml")

    #print("\n--------------------\nInitialize Monitoring\n--------------------\n")

    # Create a DataFrame
    data = {"random": [], "longest": [], "least monitored": []}

    for f in range(20):     # !!! run each output file (time steps) !!!
        f += 1
        print(f)

        runs = 1            # default 500
        for i in range(runs):  # !!! how many tests to run (how many times to rerun experiment)!!!
            #"""
            # Star broker graph
            m1 = Optical_Monitoring(broker_star, "random", True)    # random        path/cycle selection
            m2 = Optical_Monitoring(broker_star, "longest", True)   # longest       path/cycle selection
            m3 = Optical_Monitoring(broker_star, "least", True)     # least covered path/cycle selection
            dnc = {"A": 4, "B": 4, "C": 4, "D":10, "E": 4, "F": 4}
            m1.store_domain_node_count(dnc)
            m2.store_domain_node_count(dnc)
            m3.store_domain_node_count(dnc)
            #"""

            """
            # partial coop (star topology)
            m1.store_domain_info(D1, ["dA_v1", "dA_v2", "dA_v3", "dA_v4"], "A") # Domain A
            m2.store_domain_info(D1, ["dA_v1", "dA_v2", "dA_v3", "dA_v4"], "A")
            m3.store_domain_info(D1, ["dA_v1", "dA_v2", "dA_v3", "dA_v4"], "A")

            m1.store_domain_info(D2, ["dB_v1", "dB_v2", "dB_v3", "dB_v4"], "B") # Domain B
            m2.store_domain_info(D2, ["dB_v1", "dB_v2", "dB_v3", "dB_v4"], "B")
            m3.store_domain_info(D2, ["dB_v1", "dB_v2", "dB_v3", "dB_v4"], "B")

            m1.store_domain_info(D3, ["dC_v1", "dC_v2", "dC_v3", "dC_v4"], "C") # Domain C
            m2.store_domain_info(D3, ["dC_v1", "dC_v2", "dC_v3", "dC_v4"], "C")
            m3.store_domain_info(D3, ["dC_v1", "dC_v2", "dC_v3", "dC_v4"], "C")

            m1.store_domain_info(D4, ["dD_v1", "dD_v2", "dD_v3", "dD_v4", "dD_v5", "dD_v6", "dD_v7", "dD_v8", "dD_v9", "dD_v10"], "D")  # Domain D
            m2.store_domain_info(D4, ["dD_v1", "dD_v2", "dD_v3", "dD_v4", "dD_v5", "dD_v6", "dD_v7", "dD_v8", "dD_v9", "dD_v10"], "D")
            m3.store_domain_info(D4, ["dD_v1", "dD_v2", "dD_v3", "dD_v4", "dD_v5", "dD_v6", "dD_v7", "dD_v8", "dD_v9", "dD_v10"], "D")

            m1.store_domain_info(D5, ["dE_v1", "dE_v2", "dE_v3", "dE_v4"], "E") # Domain E
            m2.store_domain_info(D5, ["dE_v1", "dE_v2", "dE_v3", "dE_v4"], "E")
            m3.store_domain_info(D5, ["dE_v1", "dE_v2", "dE_v3", "dE_v4"], "E")

            m1.store_domain_info(D6, ["dF_v1", "dF_v2", "dF_v3", "dF_v4"], "F") # Domain F
            m2.store_domain_info(D6, ["dF_v1", "dF_v2", "dF_v3", "dF_v4"], "F")
            m3.store_domain_info(D6, ["dF_v1", "dF_v2", "dF_v3", "dF_v4"], "F")
            """

            # define monitoring nodes
            m = ["dA_v1", "dA_v2", "dB_v1", "dB_v2", "dB_v3", "dB_v4", "dC_v1", "dC_v2", "dC_v3", "dC_v4", "dD_v1", "dD_v2"]
            m1.add_monitoring_nodes(m)  # all nodes
            m2.add_monitoring_nodes(m)
            m3.add_monitoring_nodes(m)

            # create a trail between start and dest nodes (these can be equal -> cycle)
            for j in range(16):     # !!! number changes the amount of monitoring trails !!!
                start   = m1.select_random_monitoring_node()
                dest    = m1.select_random_monitoring_node()

                """
                # random start and end path
                m1.trail_selection(start, dest)
                m2.trail_selection(start, dest)
                m3.trail_selection(start, dest)
                """

                #"""
                # gnpy service file paths
                m1.outputfile_trail_selection(f"hard failure/gnpy path selection data/inputs/toy_3-20_rounds-40_reqs/output_file_{f}.json")
                m2.outputfile_trail_selection(f"hard failure/gnpy path selection data/inputs/toy_3-20_rounds-40_reqs/output_file_{f}.json")
                m3.outputfile_trail_selection(f"hard failure/gnpy path selection data/inputs/toy_3-20_rounds-40_reqs/output_file_{f}.json")
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
            opt = len(broker_star.edges())
            #print(f"optimal:{opt}\nscores:\n\tRandom:{score1}\n\tLongest:{score2}\n\tLeast:{score3}")
            data["random"].append(score1)
            data["longest"].append(score2)
            data["least monitored"].append(score3)

            m1, m2, m3 = None, None, None   # reset

    df = pd.DataFrame(data)

    #paths = nx.all_simple_edge_paths(physical_graph, 'dA_v1', 'dA_v5')
    #print(len(list(paths)))

    # Calculate the end time and time taken
    time_end = time.time()
    time_length = time_end - time_start

    #print(f"start time {time_start}\nend time {time_end}\ntotal time {time_length}")

    df = pd.DataFrame(data)

    # to excel file
    # df.to_excel(f'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/top3.xlsx', index=False)

    # to csv
    # df.to_csv(f'hard failure/gnpy path selection data/outputs/toy score outputs (unique paths)/top3.csv', index=False)
