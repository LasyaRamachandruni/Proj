from service_generator import GNPy_Simulator
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
	# clearing all old GNPy output files
	for filename in os.listdir("/opt/gnpy/toy-data/generated/output-result-files"):
		file_path = os.path.join("/opt/gnpy/toy-data/generated/output-result-files", filename)
		if os.path.isfile(file_path):
			os.remove(file_path)

	eqpt_file = "/opt/gnpy/eqpt/eqpt_config.json"

	# toy 2
	# topology_file = "/opt/gnpy/toy-data/toy-2-physical-topology.json"

	# toy 1
	# topology_file = "/opt/gnpy/toy-data/toy-1-physical-topology.json"
	
	# toy 3
	topology_file = "/opt/gnpy/toy-data/toy-3-physical-topology.json"


	rounds = 5
	num_req = 20

	s = GNPy_Simulator(topology_file, eqpt_file, rounds=rounds, num_req=num_req, max_ht=1)
	# s = GNPy_Simulator(topology_file, eqpt_file, rounds=rounds, num_req=num_req, single_cnxn=("dA_v1", "dD_v1"))
	s.simulate(topology_file, eqpt_file)