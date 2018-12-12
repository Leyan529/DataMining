# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import itertools
import pandas as pd


class Hits:
	def __init__(self, graph, iteration=5, min_delta=0.0001):
		self.auth = dict.fromkeys(graph.nodes, 1.0)
		self.hubs = dict.fromkeys(graph.nodes, 1.0)
		self.iteration = iteration
		self.min_delta = min_delta
		self.graph = graph

	def do(self):
		auth = self.auth
		hubs = self.hubs
		graph = self.graph
		for i in range(int(self.iteration)):

			old_auth = auth.copy()
			for node in graph.nodes:
				auth[node] = self.sum_hubs(graph, node, hubs)
			auth = self.normalization(auth)

			old_hub = hubs.copy()
			for node in graph.nodes:
				hubs[node] = self.sum_authorities(graph, node, auth)
			hubs = self.normalization(hubs)
			delta = sum((abs(old_hub[k] - hubs[k]) for k in hubs)) + sum((abs(old_auth[k] - auth[k]) for k in auth))
			if delta <= self.min_delta:
				mean_auth = sum(auth.values()) / len(auth.values())
				mean_hub = sum(hubs.values()) / len(hubs.values())
				#         print("Authority : {}".format(auth))
				#         print("Hub : {} ".format(hubs))
				print("Mean Authority : {}".format(mean_auth))
				print("Mean Hub : {}".format(mean_hub))
				return (auth, hubs, mean_auth, mean_hub)

	def sum_hubs(self, graph, node, hubs):
		s_hubs = 0.0
		if node not in graph.in_nodes:
			return s_hubs

		for in_node in graph.in_nodes[node]:
			s_hubs += hubs[in_node]
		return s_hubs

	def sum_authorities(self, graph, node, authorities):
		s_authorities = 0.0

		if node not in graph.out_nodes:
			return s_authorities

		for out_node in graph.out_nodes[node]:
			s_authorities += authorities[str(out_node)]
		return s_authorities

	def normalization(self, dic):
		norm = sum((dic[p] for p in dic))
		return {k: v / norm for (k, v) in dic.items()}

	def write_result(self, output_file):
		output_file = 'result/' + os.path.basename(output_file)
		output_file = output_file.split('.')[0]
		auth_file = output_file + '_auth.json'
		with open(auth_file, 'w') as auth_file:
			json.dump(self.auth, auth_file, indent=4)
		hub_file = output_file + '_hubs.json'
		with open(hub_file, 'w') as hub_file:
			json.dump(self.hubs, hub_file, indent=4)


class PageRank:
	def __init__(self, graph, iteration=20, damping_factor=0.85):
		self.iteration = iteration
		self.damping_factor = damping_factor
		self.ranks = None
		self.graph = graph

	def do(self):
		iteration = self.iteration
		damping_factor = self.damping_factor
		graph = self.graph
		num_nodes = len(graph.nodes)
		ranks = dict.fromkeys(graph.nodes, 1.0 / num_nodes)  # 初始化各node的rank值
		min_value = 1 - damping_factor  # (1-d)
		for i in range(int(iteration)):
			for node in graph.nodes:
				if node not in graph.out_nodes:  # no parent
					ranks[node] = 0  # 給予沒有被連接到的node=> zero rank
					continue
				rank = min_value
				for in_node in graph.in_nodes[node]:  # 計算所有鏈結至V_j的in_node
					rank += damping_factor * ranks[in_node] / len(graph.out_nodes[in_node])
				ranks[node] = rank  # PR(V_j)

		self.ranks = ranks
		#         print("Ranks : {}".format(ranks))
		return ranks

	def write_result(self, output_file):
		output_file = 'result/' + os.path.basename(output_file)
		output_file = output_file.split('.')[0]
		rank_file = output_file + '_rank.json'
		with open(rank_file, 'w') as rank_file:
			json.dump(self.ranks, rank_file, indent=4)


class SimRank:
	def __init__(self, graph, C=0.9, iteration=20):
		self.iteration = iteration
		self.C = C
		if hasattr(graph, 'wk'):
			wk = graph.wk
			self.sim = np.identity(len(wk.values()))
			self.old_sim = np.zeros(len(wk.values()))
			self.wk = {wk[k]: k for k in wk}
		else:
			self.sim = np.identity(len(graph.nodes))
			self.old_sim = np.zeros(len(graph.nodes))

		self.graph = graph

	def do(self):
		sim = self.sim
		old_sim = self.old_sim
		graph = self.graph
		for i in range(int(self.iteration)):
			old_sim = np.copy(sim)
			# 利用product笛卡爾積求多個(a,b)組合物件
			for a, b in itertools.product(graph.nodes, graph.nodes):
				#  print("a:{} , b:{} a is b :{} ".format(a,b,a == b))
				if a == b or len(graph.in_nodes[a]) == 0 or len(graph.in_nodes[b]) == 0:
					continue
				s_ab = 0  # calculate S(I_i(a),I_j(b))
				for na in graph.in_nodes[a]:
					for nb in graph.in_nodes[b]:
						s_ab += old_sim[int(na) - 1][int(nb) - 1]
				sim[int(a) - 1][int(b) - 1] = self.C / (len(graph.in_nodes[a]) * len(graph.in_nodes[b])) * s_ab
		print('pair-wise similarity of nodes...')
		size = len(old_sim[0])

		if hasattr(graph, 'wk'):
			recover_col = [self.wk[i] for i in list(range(1, size + 1))]
			self.sim = pd.DataFrame(sim, columns=recover_col, index=recover_col)
		else:
			self.sim = pd.DataFrame(sim, columns=list(range(1, size + 1)), index=list(range(1, size + 1)))
		return self.sim

	def write_result(self, output_file):
		output_file = 'result/' + os.path.basename(output_file)
		output_file = output_file.split('.')[0]
		self.sim.to_csv(output_file + '_sim_rank.csv')
