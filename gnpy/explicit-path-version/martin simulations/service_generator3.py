import json
import random
import subprocess
import os
import sys
from gnpy.topology.spectrum_assignment import frequency_to_n
from gnpy.topology.spectrum_assignment import nvalue_to_frequency
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pprint

filename = "/opt/gnpy/toy-data/generated/toy-service-file.json"

base_path_req_dic = {
    "request-id": "req_id",
    "ht": 1,
    "source": "trx_src",
    "destination": "trx_dst",
    "src-tp-id": "trx_src",
    "dst-tp-id": "trx_dst",
    "bidirectional": False,
    "path-constraints": {
        "te-bandwidth": {
            "technology": "flexi-grid",
            "trx_type": "Voyager",
            "trx_mode": "mode 1",
            "effective-freq-slot": [
                {
                  "N": None,
                  "M": None
                }
              ],
            "spacing": None,
            "path_bandwidth": 50000000000.0
        }
    }
}

base_sync_dic = {
    "synchronization-id": "1",
      "svec": {
        "relaxable": False,
        "disjointness": "node link",
        "request-id-number": ["1"]
      }
}

class GNPy_Simulator:
    def __init__(self, topology_file, eqpt_file, rounds=20, num_req=30, max_ht=5,
        single_cnxn: tuple = None, broken_fibers: list = None, loss_coef: float = None):
        self.loss_coef = loss_coef
        self.broken_fibers = broken_fibers
        self.topology_file = topology_file
        self._roadms = []
        self._trxs = []
        self._rounds = rounds
        self._num_req = num_req
        self._max_ht = max_ht
        self.single_cnxn = None
        self.paths_dir = {}
        self.use_star = False

        self.get_topology()
        self.get_eqpt(eqpt_file)
        self.create_nx_graph()
        self.create_lightpath_dict()

        if single_cnxn is not None and len(single_cnxn) == 2 and single_cnxn[0] != single_cnxn[1] and single_cnxn[0] in self._roadms and single_cnxn[1] in self._roadms:
            self.single_cnxn = single_cnxn


    def break_fiber(self, fiber_name: str):
        """
        set the loss coefficient of given fiber to loss_coef (breaking it)
        """
        data = None

        with open(self.topology_file, 'r') as file:
            data = json.load(file)
            elements = data["elements"]
            fibers = []
            for i in elements:
                if i["type"] == "Fiber" and i["uid"] == fiber_name:
                    i["params"]["loss_coef"] = self.loss_coef
        file.close()

        if data is not None:
            with open(self.topology_file, 'w') as file:
                json.dump(data, file, indent=4)
        file.close()


    def variable_break_fiber(self, fiber_name: str):
        """
        set the loss coefficient of given fiber to a variable loss (breaking it)
        """
        data = None

        with open(self.topology_file, 'r') as file:
            data = json.load(file)
            elements = data["elements"]
            fibers = []
            for i in elements:
                if i["type"] == "Fiber" and i["uid"] == fiber_name:
                    s = np.random.normal(0.2, self.loss_coef, 1)   # mean = 0.2, deviation = self.loss_coef, 1 value
                    i["params"]["loss_coef"] = s[0]
        file.close()

        if data is not None:
            with open(self.topology_file, 'w') as file:
                json.dump(data, file, indent=4)
        file.close()



    def variable_loss_coefficients(self):
        """
        update the loss coefficient of all fibers to vary slightly
        """
        data = None

        with open(self.topology_file, 'r') as file:
            data = json.load(file)
            elements = data["elements"]
            fibers = []
            for i in elements:
                if i["type"] == "Fiber":    # regular fiber
                    s = np.random.normal(0.2, 0.1, 1)   # mean = 0.2, deviation = 0.1, 1 value
                    i["params"]["loss_coef"] = s[0]
        file.close()

        if data is not None:
            with open(self.topology_file, 'w') as file:
                json.dump(data, file, indent=4)
        file.close()


    def fix_all_fibers(self):
        """
        set all loss coefficients to 0.2
        """
        data = None

        with open(self.topology_file, 'r') as file:
            data = json.load(file)
            elements = data["elements"]
            fibers = []
            for i in elements:
                if i["type"] == "Fiber" and i["params"]["loss_coef"] != 0.2:
                    i["params"]["loss_coef"] = 0.2
        file.close()

        if data is not None:
            with open(self.topology_file, 'w') as file:
                json.dump(data, file, indent=4)
        file.close()

    def view_coeff(self, f: [str]):
        """
        look at all of the loss coefficients
        """
        data = None

        with open(self.topology_file, 'r') as file:
            data = json.load(file)
            elements = data["elements"]
            fibers = []
            for i in elements:
                if (i["type"] == "Fiber") and (i["uid"] in f):
                    pprint.pprint(i)
        file.close()

        if data is not None:
            with open(self.topology_file, 'w') as file:
                json.dump(data, file, indent=4)
        file.close()


    def get_topology(self):
        """
            gets network elements (i.e. roadms) from topology file
        """
        with open(self.topology_file, 'r') as file:
            contents = file.read()
        elements = json.loads(contents)["elements"]

        for i in elements:
            if i["type"] == "Transceiver":
                self._trxs.append(i["uid"])
            if i["type"] == "Roadm":
                self._roadms.append(i["uid"])


    def get_eqpt(self, eqpt_file):
        """
            Reads minimum and maximum frequency range from equipment file.
            Also is able to see how many lightpaths are available for each roadm due to spacing.

            Make sure f_min, f_max, spacing and baud rate are the same for both the "SI" and
            "Transceiver" elemnts in the GNPy equipment file.
        """
        with open(eqpt_file, 'r') as file:
            contents = file.read()
        elements = json.loads(contents)["SI"][0]
        self._f_min = elements["f_min"]
        self._f_max = elements["f_max"]
        self._min_spacing = elements["spacing"]


    def create_nx_graph(self):
        """
            Creates networkx graph for the given topology file.
        """
        with open(self.topology_file, 'r') as file:
            contents = file.read()
        connections = json.loads(contents)["connections"]

        nodes = set()
        for i in connections:
            nodes.add(i["from_node"])
            nodes.add(i["to_node"])

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i in connections:
            G.add_edge(i["from_node"], i["to_node"])

        smaller_graph = nx.Graph()
        smaller_graph.add_nodes_from(self._roadms)
        smaller_graph.add_nodes_from(self._trxs)
        for r in self._roadms:
            for n in G.neighbors(r):
                last_seen = r
                curr = n
                while curr not in self._trxs and curr not in self._roadms:
                    temp = curr
                    curr = list((set(G.neighbors(curr)) - set([last_seen])))[0]
                    last_seen = temp
                if curr in self._trxs or curr in self._roadms:
                    smaller_graph.add_edge(r, curr)
        self.G = smaller_graph


    def create_lightpath_dict(self):
        self._l_dict = {}
        self.num_lights = int((self._f_max - self._f_min)/self._min_spacing) - 1
        for i in self._roadms:
            self._l_dict[i] = [0]*self.num_lights


    def simulate(self, topology_file, eqpt_file):
        active_cnxns = {}

        # PRINT THE GRAPH, G
        """
        # Extract unique domain identifiers and map them to colors
        domains = [node.split("_")[0][1:] for node in self.G.nodes()]
        unique_domains = list(set(domains))  # Get unique domain identifiers
        domain_to_index = {domain: i for i, domain in enumerate(unique_domains)}  # Map domains to indices

        # Create colormap and assign colors
        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0, vmax=len(unique_domains) - 1)
        node_colors = [cmap(norm(domain_to_index[domain])) for domain in domains]

        # Draw the graph
        nx.draw(self.G, node_color=node_colors, with_labels=True)
        plt.savefig('foo.png')
        """

        for r in range(self._rounds):
            # decrementing holding time by 1
            reqs_to_delete = []
            for req_id in active_cnxns.keys():
                active_cnxns[req_id]["remaining_time"] -= 1

                # if service has expired, remove from active connections
                if active_cnxns[req_id]["remaining_time"] == 0:
                    my_light = active_cnxns[req_id]["chosen_light"]
                    for curr_roadm in active_cnxns[req_id]["chosen_path"]:
                        self._l_dict[curr_roadm][my_light] = 0
                    reqs_to_delete.append(req_id)
            for i in reqs_to_delete:
                del active_cnxns[i]

            # for the services still active from the previous time round, we will add them to
            # this time round's service file
            prev_services_dict = {}
            if len(active_cnxns) > 0:
                with open(filename, 'r') as file:
                    contents = file.read()
                prev_services_list = json.loads(contents)["path-request"]
                for i in prev_services_list:
                    i["ht"] -= 1
                    prev_services_dict[i["request-id"]] = i

            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, "x") as f:
                f.write("{\n\t\"path-request\": [\n")

            for k in active_cnxns.keys():
                with open(filename, "a") as f:
                    json.dump(prev_services_dict[k], f, indent=4)  # indent for pretty formatting
                    f.write(",\n")

            # fixed paths instead of n random
            self.fixed_paths = {
                ('trx_dA_v2', 'trx_dC_v1'): [[('trx_dA_v2', 'dA_v2'), ('dA_v2', 'dB_v3'), ('dB_v3', 'dB_v4'), ('dB_v4', 'dB_v2'), ('dB_v2', 'dB_v5'), ('dB_v5', 'dB_v1'), ('dB_v1', 'dA_v1'), ('dA_v1', 'dC_v1'), ('dC_v1', 'trx_dC_v1')]],                                                                                                                                             # red
                ('trx_dD_v2', 'trx_dC_v1'): [[('trx_dD_v2', 'dD_v2'), ('dD_v2', 'dC_v2'), ('dC_v2', 'dC_v4'), ('dC_v4', 'dC_v3'), ('dC_v3', 'dA_v2'), ('dA_v2', 'dA_v4'), ('dA_v4', 'dA_v3'), ('dA_v3', 'dA_v1'), ('dA_v1', 'dC_v1'), ('dC_v1', 'trx_dC_v1')]],                                                                                                                         # pink
                ('trx_dC_v3', 'trx_dC_v1'): [[('trx_dC_v3', 'dC_v3'), ('dC_v3', 'dA_v2'), ('dA_v2', 'dA_v4'), ('dA_v4', 'dA_v3'), ('dA_v3', 'dA_v1'), ('dA_v1', 'dB_v1'), ('dB_v1', 'dB_v5'), ('dB_v5', 'dB_v2'), ('dB_v2', 'dB_v4'), ('dB_v4', 'dD_v1'), ('dD_v1', 'dD_v3'), ('dD_v3', 'dD_v2'), ('dD_v2', 'dC_v2'), ('dC_v2', 'dC_v5'), ('dC_v5', 'dC_v1'), ('dC_v1', 'trx_dC_v1')]], # orange
                ('trx_dA_v1', 'trx_dC_v1'): [[('trx_dA_v1', 'dA_v1'), ('dA_v1', 'dA_v3'), ('dA_v3', 'dA_v4'), ('dA_v4', 'dA_v2'), ('dA_v2', 'dB_v3'), ('dB_v3', 'dB_v4'), ('dB_v4', 'dB_v2'), ('dB_v2', 'dD_v1'), ('dD_v1', 'dD_v3'), ('dD_v3', 'dD_v2'), ('dD_v2', 'dC_v2'), ('dC_v2', 'dC_v4'), ('dC_v4', 'dC_v3'), ('dC_v3', 'dC_v5'), ('dC_v5', 'dC_v1'), ('dC_v1', 'trx_dC_v1')]], # green
                ('trx_dC_v4', 'trx_dD_v1'): [[('trx_dC_v4', 'dC_v4'), ('dC_v4', 'dC_v2'), ('dC_v2', 'dC_v5'), ('dC_v5', 'dC_v1'), ('dC_v1', 'dA_v1'), ('dA_v1', 'dA_v3'), ('dA_v3', 'dA_v4'), ('dA_v4', 'dA_v2'), ('dA_v2', 'dB_v3'), ('dB_v3', 'dB_v4'), ('dB_v4', 'dB_v2'), ('dB_v2', 'dD_v1'), ('dD_v1', 'trx_dD_v1')]],                                                             # blue
                ('trx_dC_v2', 'trx_dD_v1'): [[('trx_dC_v2', 'dC_v2'), ('dC_v2', 'dC_v4'), ('dC_v4', 'dC_v3'), ('dC_v3', 'dA_v2'), ('dA_v2', 'dA_v4'), ('dA_v4', 'dA_v3'), ('dA_v3', 'dA_v1'), ('dA_v1', 'dB_v1'), ('dB_v1', 'dB_v5'), ('dB_v5', 'dB_v2'), ('dB_v2', 'dB_v4'), ('dB_v4', 'dD_v1'), ('dD_v1', 'trx_dD_v1')]]                                                              # gray
            }

            # for the remaining available service slots, we create new ones
            fix_path_len = len(self.fixed_paths)
            for n in range(self._num_req - len(active_cnxns)):

                (src, des) = list(self.fixed_paths.keys())[n%fix_path_len]  # get from hard coded values
                # print(f"{n%4}  {list(self.fixed_paths.keys())[n%fix_path_len]}")
                my_dict = dict(base_path_req_dic)

                """
                # old src / dest pair logic from Soham
                if self.use_star:
                    src = random.choice(self._trxs)
                    des = random.choice(self._trxs)
                    while src == des or src not in self.star_trxs or des not in self.star_trxs:
                        src = random.choice(self._trxs)
                        des = random.choice(self._trxs)
                elif self.single_cnxn is None:
                    src = random.choice(self._trxs)
                    des = random.choice(self._trxs)
                    while src == des:
                        des = random.choice(self._trxs)
                else:
                    src = "trx_" + self.single_cnxn[0]
                    des = "trx_" + self.single_cnxn[1]
                """
                """
                # old path logic from Soham
                # now we check if any lightpath can accomodate this request
                possible_paths = None
                if (min(src, des), max(src, des)) not in self.paths_dir:
                    if self.use_star:
                        self.paths_dir[(min(src, des), max(src, des))] = sorted(list(nx.all_simple_edge_paths(self.star_g, src, des, cutoff=7)))
                    else:
                        self.paths_dir[(min(src, des), max(src, des))] = sorted(list(nx.all_simple_edge_paths(self.G, src, des, cutoff=7)))
                possible_paths = self.paths_dir[(min(src, des), max(src, des))]
                print("Path count", len(possible_paths), len(self.paths_dir))
                """
                possible_paths = self.fixed_paths[(src, des)]

                chosen_path = None
                chosen_light = None
                # print(f"Round {n} \n------------------\n\n")
                for p in possible_paths:
                    # deleting trx nodes from path and center node (in case of star graph)
                    # print(p, "\n")

                    roadms_in_path = []
                    for (a,b) in p:
                        if a in self._roadms: roadms_in_path.append(a)

                    # reversing list (if needed)
                    if src[4:] != roadms_in_path[0]:
                        roadms_in_path.reverse()

                    # find available lights in current path
                    total_light_count = [0]*self.num_lights
                    for curr_roadm in roadms_in_path:
                        for i in range(len(total_light_count)):
                            total_light_count[i] += self._l_dict[curr_roadm][i] # should add either 1 or 0 (used/unused)
                    for i in range(len(total_light_count)):
                        if total_light_count[i] == 0:
                            # we found a light that is unused by all roadms in this path
                            chosen_light = i
                            chosen_path = roadms_in_path
                            for curr_roadm in chosen_path:
                                if self._l_dict[curr_roadm][chosen_light] == 1:
                                    raise Exception("Already in-use light!")
                                self._l_dict[curr_roadm][chosen_light] = 1
                            break
                    if chosen_light is not None:
                        break
                if chosen_path is None:
                    continue

                # now we have our desired lightpath
                N = freq_to_N(self._f_min + ((chosen_light + 1) * self._min_spacing)) + 1
                M = width_to_M(self._min_spacing)

                my_dict["request-id"] = str(r+1) + "_" + str(n+1)
                my_dict["source"] = src
                my_dict["src-tp-id"] = src
                my_dict["destination"] = des
                my_dict["dst-tp-id"] = des
                my_dict["path-constraints"]["te-bandwidth"]["effective-freq-slot"][0]["N"] = N
                my_dict["path-constraints"]["te-bandwidth"]["effective-freq-slot"][0]["M"] = M
                my_dict["path-constraints"]["te-bandwidth"]["spacing"] = self._min_spacing
                # my_dict["ht"] = random.randint(1, self._max_ht)
                my_dict["ht"] = self._max_ht

                # following block explicitly specifies route of this connection
                my_dict["explicit-route-objects"] = {}
                my_dict["explicit-route-objects"]["route-object-include-exclude"] = []
                for i in range(len(chosen_path)):
                    curr = {}
                    curr["explicit-route-usage"] = "route-include-ero"
                    curr["index"] = i
                    curr["num-unnum-hop"] = {}
                    curr["num-unnum-hop"]["node-id"] = chosen_path[i]
                    curr["num-unnum-hop"]["hop-type"] = "STRICT"
                    my_dict["explicit-route-objects"]["route-object-include-exclude"].append(curr)


                # adding to active connection log
                active_cnxns[my_dict["request-id"]] = {}
                active_cnxns[my_dict["request-id"]]["chosen_path"] = chosen_path
                active_cnxns[my_dict["request-id"]]["chosen_light"] = chosen_light
                active_cnxns[my_dict["request-id"]]["remaining_time"] = my_dict["ht"]

                with open(filename, "a") as f:
                    json.dump(my_dict, f, indent=4)  # indent for pretty formatting
                    f.write(",\n")

            # removing trailing comma
            with open(filename, 'rb+') as filehandle:
                filehandle.seek(-2, os.SEEK_END)
                filehandle.truncate()

            # adding sync
            with open(filename, "a") as f:
                f.write("],\n\"synchronization\": [\n")
                my_dict = dict(base_sync_dic)
                my_dict["svec"]["request-id-number"] = [list(active_cnxns.keys())[0]]
                json.dump(my_dict, f, indent=4)
                f.write("\n\t]\n}")

            # breaking a fiber
            if self.broken_fibers is not None and self.loss_coef is not None:

                self.variable_loss_coefficients()
                if random.randint(0,100) <= 20:  # 20% chance to break fiber
                    chosen_broken_fiber = random.choice(self.broken_fibers)
                    self.break_fiber(chosen_broken_fiber)
                    # self.view_coeff(self.broken_fibers)

                    subdirectory_path = "/opt/gnpy/toy-data/generated/output-result-files/" + chosen_broken_fiber.replace("/", " of ")
                    # Create the subdirectory (and any necessary parent directories)
                    try:
                        os.makedirs(subdirectory_path)
                    except FileExistsError:
                        # do nothing
                        print()
                    except Exception as e:
                        sys.exit(1)

                    # simulating with failure
                    subprocess.run(["gnpy-path-request", "-o", subdirectory_path + f"/output_file_{r+1}.json",
                        "-e", eqpt_file, self.topology_file, filename])
                else:   # 80% chance of simulating regular fiber health

                    subdirectory_path = "/opt/gnpy/toy-data/generated/output-result-files/regular_traffic"
                    # Create the subdirectory (and any necessary parent directories)
                    try:
                        os.makedirs(subdirectory_path)
                    except FileExistsError:
                        # do nothing
                        print()
                    except Exception as e:
                        sys.exit(1)

                    # simulating with variable loss
                    subprocess.run(["gnpy-path-request", "-o", subdirectory_path + f"/output_file_{r+1}.json",
                        "-e", eqpt_file, self.topology_file, filename])


                if self.broken_fibers is not None:
                    self.fix_all_fibers()

                # simulating with no failure (control)
                subprocess.run(["gnpy-path-request", "-o", f"/opt/gnpy/toy-data/generated/output-result-files/output_file_{r+1}.json",
                    "-e", eqpt_file, self.topology_file, filename])
            else:
                '''
                subprocess.run(["gnpy-path-request", "-o", f"/opt/gnpy/toy-data/generated/output-result-files/output_file_{r+1}.json",
                    "-e", eqpt_file, self.topology_file, filename], stdout=subprocess.DEVNULL)
                '''
                subprocess.run(["gnpy-path-request", "-o", f"/opt/gnpy/toy-data/generated/output-result-files/output_file_{r+1}.json",
                    "-e", eqpt_file, self.topology_file, filename])

def width_to_M(f_val):
    return int(f_val / 1000000000.0 / 12.5)

def freq_to_N(w_val):
    return frequency_to_n(w_val)

def M_to_width(M: int):
    return 12.5 * M * 1000000000.0

def N_to_freq(N: int):
    return nvalue_to_frequency(N)
