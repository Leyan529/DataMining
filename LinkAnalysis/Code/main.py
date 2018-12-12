# -*- coding: utf-8 -*-

import numpy as np
import operator
import datetime
from GP import TCGraph, Graph
from Algorithm import Hits, PageRank, SimRank


def linkTest(fn, itr):
	fmt = fn.split('.')[1]
	if fmt != 'csv':
		graph = Graph(fn)
	else:
		graph = TCGraph(fn)
		wk = {graph.wk[k]: k for k in graph.wk}
	# print(wk)

	s_hits = datetime.datetime.now()
	hits = Hits(graph, iteration=itr, min_delta=0.0001)
	auth, hubs, _, _ = hits.do()
	e_hits = datetime.datetime.now()
	print("Do hits during time : {} ms".format((e_hits - s_hits).microseconds))
	hits.write_result(fn)

	# 大到小sort
	sorted_auth = sorted(auth.items(), key=operator.itemgetter(1), reverse=True)
	sorted_hubs = sorted(hubs.items(), key=operator.itemgetter(1), reverse=True)
	if hasattr(graph, 'wk'):
		print('Top 10 sorted_auth : {}'.format([wk[int(a[0])] for a in sorted_auth[:5]]))
		print('Top 10 sorted_hubs : {}'.format([wk[int(a[0])] for a in sorted_hubs[:5]]))
	else:
		print('Top 10 sorted_auth : {}'.format([a[0] for a in sorted_auth[:5]]))
		print('Top 10 sorted_hubs : {}'.format([a[0] for a in sorted_hubs[:5]]))

	# 小到大sort
	sorted_auth = sorted(auth.items(), key=operator.itemgetter(1))
	sorted_hubs = sorted(hubs.items(), key=operator.itemgetter(1))
	if hasattr(graph, 'wk'):
		print('Reciprocal 10 sorted_auth : {}'.format([wk[int(a[0])] for a in sorted_auth[:5]]))
		print('Reciprocal 10 sorted_hubs : {}'.format([wk[int(a[0])] for a in sorted_hubs[:5]]))
	else:
		print('Reciprocal 10 sorted_auth : {}'.format([a[0] for a in sorted_auth[:5]]))
		print('Reciprocal 10 sorted_hubs : {}'.format([a[0] for a in sorted_hubs[:5]]))

	# page_rank
	# 6 graphs in project3dataset
	s_rank = datetime.datetime.now()
	page_rank = PageRank(graph, iteration=itr, damping_factor=0.85)
	ranks = page_rank.do()
	e_rank = datetime.datetime.now()
	print("Do page_rank during time : {} ms".format((e_rank - s_rank).microseconds))
	page_rank.write_result(fn)

	# 大到小sort
	sorted_rank = sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)
	if hasattr(graph, 'wk'):
		print('sorted_rank : {}'.format([wk[int(a[0])] for a in sorted_rank[:5]]))
	else:
		print('sorted_rank : {}'.format([a[0] for a in sorted_rank[:5]]))

	# 小到大sort
	sorted_rank = sorted(ranks.items(), key=operator.itemgetter(1))
	if hasattr(graph, 'wk'):
		print('Reciprocal sorted_rank : {}'.format([wk[int(a[0])] for a in sorted_rank[:5]]))
	else:
		print('Reciprocal sorted_rank : {}'.format([a[0] for a in sorted_rank[:5]]))

	# sim_rank
	# first 5 graphs of project3dataset.
	s_sim = datetime.datetime.now()
	simRank = SimRank(graph, C=0.9, iteration=itr)
	sim = simRank.do()
	e_sim = datetime.datetime.now()
	print("Do sim_rank during time : {} ms".format((e_sim - s_sim).microseconds))
	simRank.write_result(fn)

	#     print('---Similarity Orignal matrix---')
	#     print(np.matrix(sim))

	matrix = np.matrix(sim.replace(1, 0))
	max_score = matrix.max()
	print("Node Max Similarity value (Without self) : {} ".format(max_score))

	inverse_unique_set = []
	for i in range(0, len(matrix)):
		for j in range(0, len(matrix)):
			if (sim.iloc[i, j] == max_score and (max_score != 0) and (i + 1, j + 1) not in inverse_unique_set):
				if hasattr(graph, 'wk'):
					print("Node:{} & Node:{} Similarity high : {}".format(wk[(i + 1)], wk[(j + 1)], max_score))
				else:
					print("Node:{} & Node:{} Similarity high : {}".format(i + 1, j + 1, max_score))
				inverse_unique_set.append((j + 1, i + 1))
				break


if __name__ == '__main__':

	for i in range(1, 5):
		fn = 'hw3dataset/graph_{}.txt'.format(i)
		linkTest(fn, itr=20)
		print('\n' + '---------------' + fn + 'Over' + '---------------' + '\n')

	for i in range(5, 7):
		fn = 'hw3dataset/graph_{}.txt'.format(i)
		linkTest(fn, itr=50)
		print('\n' + '---------------' + fn + 'Over' + '---------------' + '\n')

	fn = 'hw3dataset/BreadBasket_DMS.csv'
	linkTest(fn, itr=20)
	print('\n' + '---------------' + fn + 'Over' + '---------------' + '\n')
