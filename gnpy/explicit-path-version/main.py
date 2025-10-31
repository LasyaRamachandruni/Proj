from service_generator import GNPy_Simulator
import matplotlib.pyplot as plt
import os
import subprocess

if __name__ == "__main__":
	# clearing all old GNPy output files
	'''
	for filename in os.listdir(""):
		file_path = os.path.join("/opt/gnpy/toy-data/generated/output-result-files", filename)
		if os.path.isfile(file_path):
			os.remove(file_path)
	'''

	subprocess.Popen('rm -rf /opt/gnpy/toy-data/generated/output-result-files', shell=True)
	subprocess.Popen('mkdir -p /opt/gnpy/toy-data/generated/output-result-files', shell=True)

	eqpt_file = "/opt/gnpy/eqpt/eqpt_config.json"

	# toy 2
	topology_file = "/opt/gnpy/toy-data/toy-2-physical-topology.json"
	broken_fibers = [
		"fiber (dA_v1 \u2192 dC_v1)_(2/2)",
		"fiber (dD_v2 \u2192 dC_v2)_(2/2)",
		"fiber (dB_v4 \u2192 dD_v1)_(2/2)",
		"fiber (dA_v2 \u2192 dB_v3)_(1/2)"
	]

	# toy 1
	# topology_file = "/opt/gnpy/toy-data/toy-1-physical-topology.json"

	# toy 3
	# topology_file = "/opt/gnpy/toy-data/toy-3-physical-topology.json"

	rounds = 100
	num_req = 44

	s = GNPy_Simulator(topology_file, eqpt_file, rounds=rounds, num_req=num_req, max_ht=100, broken_fibers=broken_fibers,
		loss_coef=0.8, fail_proportion=0.2)
	s.simulate(topology_file, eqpt_file)