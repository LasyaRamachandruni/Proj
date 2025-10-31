import json
import random
import subprocess
import os
from gnpy.topology.spectrum_assignment import frequency_to_n
from gnpy.topology.spectrum_assignment import nvalue_to_frequency
import networkx as nx

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
    def __init__(self, topology_file, eqpt_file, rounds=20, num_req=30, max_ht=5, single_cnxn: tuple = None):
        self._roadms = []
        self._trxs = []
        self._rounds = rounds
        self._num_req = num_req
        self._max_ht = max_ht
        self.single_cnxn = None
        self.paths_dir = {}

        self.get_topology(topology_file)
        self.get_eqpt(eqpt_file)
        self.create_nx_graph(topology_file)

        if single_cnxn is not None and len(single_cnxn) == 2 and single_cnxn[0] != single_cnxn[1] and single_cnxn[0] in self._roadms and single_cnxn[1] in self._roadms:
            self.single_cnxn = single_cnxn

    def get_topology(self, topology_file):
        """
            gets network elements (i.e. roadms) from topology file
        """
        with open(topology_file, 'r') as file:
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

    def create_nx_graph(self, topology_file):
        """
            Creates networkx graph for the given topology file.
        """
        with open(topology_file, 'r') as file:
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

    def simulate(self, topology_file, eqpt_file):
        for r in range(self._rounds):
            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, "x") as f:
                f.write("{\n\t\"path-request\": [\n")

            # for the remaining available service slots, we create new ones
            for n in range(self._num_req):
                my_dict = dict(base_path_req_dic)


                if self.single_cnxn is None:
                    src = random.choice(self._trxs)
                    des = random.choice(self._trxs)
                    while src == des:
                        des = random.choice(self._trxs)
                else:
                    src = "trx_" + self.single_cnxn[0]
                    des = "trx_" + self.single_cnxn[1]
                
                my_dict["request-id"] = str(r+1) + "_" + str(n+1)
                my_dict["source"] = src
                my_dict["src-tp-id"] = src
                my_dict["destination"] = des
                my_dict["dst-tp-id"] = des
                my_dict["path-constraints"]["te-bandwidth"]["spacing"] = self._min_spacing
                my_dict["ht"] = random.randint(1, self._max_ht)

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
                my_dict["svec"]["request-id-number"] = [str(r+1) + "_1"]
                json.dump(my_dict, f, indent=4)
                f.write("\n\t]\n}")
            '''
            subprocess.run(["gnpy-path-request", "-o", f"/opt/gnpy/toy-data/generated/output-result-files/output_file_{r+1}.json", 
                "-e", eqpt_file, topology_file, filename], stdout=subprocess.DEVNULL)
            '''
            subprocess.run(["gnpy-path-request", "-o", f"/opt/gnpy/toy-data/generated/output-result-files/output_file_{r+1}.json", 
                "-e", eqpt_file, topology_file, filename])

def width_to_M(f_val):
    return int(f_val / 1000000000.0 / 12.5)

def freq_to_N(w_val):
    return frequency_to_n(w_val)
        
def M_to_width(M: int):
    return 12.5 * M * 1000000000.0

def N_to_freq(N: int):
    return nvalue_to_frequency(N)