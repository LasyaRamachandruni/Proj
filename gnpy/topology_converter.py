import networkx as nx
import json

roadm_dict = {
			"uid": None,
			"type": "Roadm",
			"params": {
				"target_pch_out_db": -20,
				"restrictions": {
				  "preamp_variety_list": [
					"openroadm_mw_mw_preamp"
				  ],
				  "booster_variety_list": [
					"openroadm_mw_mw_booster"
				  ]
				},
				"per_degree_pch_out_db": None
			}
		}

fiber_dict = {
		# "fiber (C2 → C3)_(2/3)"
		  "uid": None,
		  "type": "Fiber",
		  "type_variety": "SSMF",
		  "params": {
			"length": 50.0,
			"loss_coef": 0.2,
			"length_units": "km",
			"att_in": 0,
			"con_in": 0,
			"con_out": 0
		  }
		}

booster_dict = {
		  # "Edfa_booster_roadm_A1_to_fiber (A1 → A4)_(1/2)"
		  "uid": None,
		  "type": "Edfa",
		  "type_variety": "openroadm_mw_mw_booster",
		  "operational": {
			"gain_target": 15.0,
			"delta_p": 0,
			"tilt_target": 0,
			"out_voa": 0
		  }
		}

preamp_dict = {
		  # "Edfa_preamp_roadm_C1_from_fiber (A1 → C1)"
		  "uid": None,
		  "type": "Edfa",
		  "type_variety": "openroadm_mw_mw_preamp",
		  "operational": {
			"gain_target": 15.0,
			"delta_p": 0,
			"tilt_target": 0,
			"out_voa": 0
		  }
		}

edfa_fiber_dict = {
		  # "Edfa_fiber (A4 → A1)_(1/2)"
		  "uid": None,
		  "type": "Edfa",
		  "type_variety": "openroadm_ila_low_noise",
		  "operational": {
			"gain_target": 15.0,
			"delta_p": 0,
			"tilt_target": 0,
			"out_voa": 0
		  }
		}

def convert(g: nx.classes.graph.Graph):
	elements = []
	connections = []
	elements_per_roadm = {}

	roadm_ids = g.nodes
	transceivers_ids = ['trx_' + i for i in roadm_ids]

	# adding transceivers to elements
	for i in transceivers_ids:
		elements.append({"uid": i, "type": "Transceiver"})
		connections.append({"from_node": i,"to_node": i[4:]})
		connections.append({"from_node": i[4:], "to_node": i})

	# adding roadms to elements
	for i in roadm_ids:
		# updating dictionary of components
		elements_per_roadm[i] = {}
		for n in list(g.neighbors(i)):
			elements_per_roadm[i][n] = {"fiber": [], "Edfa_fiber": [], "Edfa_booster": "", "Edfa_preamp": ""}

		print(i,': ',list(g.neighbors(i)))
		curr_dict = dict(roadm_dict)
		curr_dict["uid"] = i
		curr_dict["params"]["per_degree_pch_out_db"] = {}

		for e in g.neighbors(i):
			add_on = ''
			if g.get_edge_data(i, e)["weight"] > 1:
				add_on = '_(1/' + str(int(g.get_edge_data(i, e)["weight"])) + ')'
			curr_str = 'Edfa_booster_'+i+'_to_fiber ('+i+' → '+e+')'+add_on
			elements_per_roadm[i][e]["Edfa_booster"] = curr_str
			print(curr_str)

			# for sub-field in roadm dict
			curr_dict["params"]["per_degree_pch_out_db"][curr_str] = -20
		elements.append(curr_dict)

		for e in list(g.neighbors(i)):
			add_on = ''
			if g.get_edge_data(i, e)["weight"] > 1:
				add_on = '_(1/' + str(int(g.get_edge_data(i, e)["weight"])) + ')'
			curr_str = 'Edfa_booster_'+i+'_to_fiber ('+i+' → '+e+')'+add_on

			# for booster dict
			curr_booster_dict = dict(booster_dict)
			curr_booster_dict['uid'] = curr_str
			elements.append(curr_booster_dict)

		for e in list(g.neighbors(i)):
			add_on = ''
			if g.get_edge_data(i, e)["weight"] > 1:
				add_on = '_('+ str(int(g.get_edge_data(i, e)["weight"])) +'/' + str(int(g.get_edge_data(i, e)["weight"])) + ')'
			curr_str = 'Edfa_preamp_'+i+'_from_fiber ('+e+' → '+i+')'+add_on
			elements_per_roadm[i][e]["Edfa_preamp"] = curr_str

			# for preamp dict
			curr_preamp_dict = dict(preamp_dict)
			curr_preamp_dict['uid'] = curr_str
			elements.append(curr_preamp_dict)

		for e in list(g.neighbors(i)):
			if g.get_edge_data(i, e)["weight"] > 1:
				for c in range(1, g.get_edge_data(i, e)["weight"]):
					add_on = '_('+ str(c) +'/' + str(int(g.get_edge_data(i, e)["weight"])) + ')'
					curr_str = 'Edfa_fiber ('+i+' → '+e+')'+add_on
					elements_per_roadm[i][e]["Edfa_fiber"].append(curr_str)

					# for edfa fiber dict
					curr_edfa_fiber_dict = dict(edfa_fiber_dict)
					curr_edfa_fiber_dict['uid'] = curr_str
					elements.append(curr_edfa_fiber_dict)

	# adding fibers to elements
	for i in roadm_ids:
		for n in g.neighbors(i):
			weight = g.get_edge_data(i, n)["weight"]
			uid = "fiber (" + i + " → " + n + ")"
			if weight > 1:
				for w in range(1, weight + 1):
					curr_dict = dict(fiber_dict)
					curr_dict['uid'] = uid + '_(' + str(w) + '/' + str(weight) + ')'
					elements_per_roadm[i][n]["fiber"].append(curr_dict['uid'])
					elements.append(curr_dict)
			else:
				curr_dict = dict(fiber_dict)
				curr_dict['uid'] = uid
				elements_per_roadm[i][n]["fiber"].append(curr_dict['uid'])
				elements.append(curr_dict)


	# Connections
	for i in roadm_ids:
		for n in g.neighbors(i):
			ordered_path = []
			curr_dict = elements_per_roadm[i][n]
			# add source roadm
			ordered_path.append(i)

			# add booster
			ordered_path.append(curr_dict["Edfa_booster"])

			# add fibers and edfas
			for j in range(len(curr_dict["fiber"]) - 1):
				ordered_path.append(curr_dict["fiber"][j])
				ordered_path.append(curr_dict["Edfa_fiber"][j])
			ordered_path.append(curr_dict["fiber"][len(curr_dict["fiber"]) - 1])

			# add preamp
			ordered_path.append(curr_dict["Edfa_preamp"])

			# add destination roadm
			ordered_path.append(n)

			for j in range(len(ordered_path) - 1):
				connections.append({
				  "from_node": ordered_path[j],
				  "to_node": ordered_path[j+1]
				})

	retval_dict = {
		"elements": elements,
		"connections": connections
	}

	json_string = json.dumps(retval_dict, indent=4)

	with open("/Users/soham/Documents/Coding-Projects/optical-projects/gnpy/toy-data/toy-3-star-topology.json", "w") as f:
		f.write(json_string)