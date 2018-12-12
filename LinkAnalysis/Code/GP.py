# -*- coding: utf-8 -*-
import re
import copy
import itertools
import pandas as pd


class Graph:
	def __init__(self, f):
		self.graph, self.nodes = self.read_graph(open(f))
		self.in_nodes = self.get_in_nodes()
		self.out_nodes = self.graph

	def get_in_nodes(self):
		in_nodes = {}
		for node in self.nodes:
			in_nodes[node] = []

		for source, targets in self.graph.items():
			for node in targets:
				if str(source) not in in_nodes[str(node)]:
					in_nodes[str(node)].append(str(source))
		return in_nodes

	def read_graph(self, f):
		graph = {}
		nodes = []
		for line in f.readlines():
			source, target = self.parse_format(line)
			if source not in graph:
				graph[source] = [target]
			else:
				graph[source].append(target)

			if source not in nodes:
				nodes.append(source)
			if target not in nodes:
				nodes.append(target)
		return graph, nodes

	def parse_format(self, line):
		source, target = line.split(',')
		#         match = re.search('([0-9]+)', target)
		match = re.search('([0-9A-Za-z]+)', target)  # 擴充通用
		if match:
			target = match.groups()[0]
		return source, target


class TCGraph:
	def __init__(self, f):
		print('transform dataset...')
		TDB_List = self.read_transaction(f)
		f = f.replace('csv', 'txt')
		print('making graph...')
		self.make_graph(copy.deepcopy(TDB_List), f)
		self.graph, self.nodes = self.read_graph(open(f))
		self.in_nodes = self.get_in_nodes()
		self.out_nodes = self.graph

	def get_in_nodes(self):
		in_nodes = {}
		for node in self.nodes:
			in_nodes[node] = []

		for source, targets in self.graph.items():
			for node in targets:
				if str(source) not in in_nodes[str(node)]:
					in_nodes[str(node)].append(str(source))
		return in_nodes

	def read_graph(self, f):
		graph = {}
		nodes = []
		for line in f.readlines():
			source, target = self.parse_format(line)
			if source not in graph:
				graph[source] = [target]
			else:
				graph[source].append(target)

			if source not in nodes:
				nodes.append(source)
			if target not in nodes:
				nodes.append(target)
		return graph, nodes

	def read_transaction(self, f):
		bakery_data = pd.read_csv(f, encoding='utf-8')
		bakery_data = bakery_data.drop(['Date', 'Time'], axis=1)
		bakery_data = bakery_data[~bakery_data['Item'].str.contains('NONE')]
		wk = self.wordKey(bakery_data['Item'].unique())

		tdl = []
		for i in range(1, bakery_data.Transaction.count() + 1):
			tdf = bakery_data[bakery_data.Transaction == i]
			l = set()
			for j in range(0, tdf.Transaction.count()):
				l.add(wk[tdf.Item.iloc[j]])
			if len(l) > 0:
				tdl.append(list(l))
			else:
				tdl.append(None)

		col = ['items']
		TDB = pd.DataFrame({"items": tdl}, columns=col)
		TDB = TDB.dropna()
		return TDB['items'].tolist()

	def make_graph(self, tdb, wfn):
		edge_list = []
		wk = {self.wk[k]: k for k in self.wk}
		for t in tdb:
			for a, b in itertools.product(t, t):
				if a != b and ((b, a) not in edge_list) and ((a, b) not in edge_list):
					#                     edge_list.append((wk[int(a)],wk[int(b)])) # 文字版
					edge_list.append((a, b))  # 數字版

		print('edge finished')
		with open(wfn, 'w') as f:
			for e in edge_list:
				line = str(e).replace('(', '').replace(')', '').replace("'", '')
				f.write(line)
				f.write('\n')

	def parse_format(self, line):
		source, target = line.split(',')
		#         match = re.search('([0-9]+)', target)
		match = re.search('([0-9]+)', target)  # 擴充通用
		if match:
			target = match.groups()[0]
		return source, target

	def wordKey(self, items):
		# 轉化字典
		node_list = pd.Series(items).astype('category').cat.codes.values + 1
		word_key = {}
		for i in range(0, len(node_list)):
			word_key[items[i]] = node_list[i]
		print('-------------word_key--------------')
		self.wk = word_key
		#         print(word_key)
		return word_key
