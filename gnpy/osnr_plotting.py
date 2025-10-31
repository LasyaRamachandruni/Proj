import matplotlib.pyplot as plt
import os
import json

vals = []
rounds = 10

'''
for i in range(1, rounds + 1):
	curr_filename = f"/opt/gnpy/toy-data/generated/output-result-files/output_file_{i}.json"

	total_osnr = 0
	num_services = 0

	with open(curr_filename, 'r') as file:
		# Load the JSON data into a Python dictionary
		responses = json.load(file)['response']

	for i in responses:
		if "path-properties" not in i:
			continue

		num_services += 1
		path_metric = i["path-properties"]["path-metric"]
		for pm in path_metric:
			if pm["metric-type"] == "SNR-bandwidth":
				total_osnr += pm["accumulative-value"]

	vals.append(total_osnr/num_services)
'''

link_utils = {
"High Spacing":[23.808571428571433, 23.119, 23.223636363636363, 23.885, 23.75384615384616, 23.2625, 22.769166666666667, 22.91, 22.967142857142857, 23.83923076923077]
,
"Minimal Spacing":[18.468888888888888, 18.21055555555556, 17.952222222222222, 17.742631578947368, 17.776500000000006, 17.35625, 18.06833333333333, 18.311052631578946, 18.549473684210522, 17.871052631578948]
}

x = list(range(1, rounds+1))
for k in link_utils.keys():
	plt.plot(x, link_utils[k], label=k)
plt.legend()
plt.xlabel("Round")
plt.ylabel("SNR")
plt.savefig('/opt/gnpy/toy-data/generated/plots/osnr-plot.png') 