#
import copy
import time
import sys
import pandas as pd
import numpy as np
import re

from collections import defaultdict
import csv


class Apriori(object):
	def __init__(self, minSupp, minConf):
		""" Parameters setting
		"""
		self.minSupp = minSupp  # min support (used for mining frequent sets)
		self.minConf = minConf  # min confidence (used for mining association rules)

	def fit(self, data):
		""" Run the apriori algorithm, return the frequent *-term sets.
		"""
		# Initialize some variables to hold the tmp result
		transListSet = data  # 取得transaction list
		itemSet = self.getOneItemSet(transListSet)  # get 1-item set
		itemCountDict = defaultdict(int)  # key=candiate k-item(k=1/2/...), value=count ,為不存在的 key 設定預設值int
		freqSet = dict()  # 儲存所有的 frequent *-items set

		self.transLength = len(transListSet)  # number of transactions
		self.itemSet = itemSet

		# Get the frequent 1-term set
		freqOneTermSet = self.getItemsWithMinSupp(transListSet, itemSet, itemCountDict, self.minSupp)  # L1

		# Main loop
		k = 1
		currFreqTermSet = freqOneTermSet  # 當前從L1開始計算
		while currFreqTermSet != set():  # 假設當前Lk不為空則迭代進行計算
			freqSet[k] = currFreqTermSet  # 儲存Lk items set
			k += 1  # k提升一階
			currCandiItemSet = self.getJoinedItemSet(currFreqTermSet, k)  # get new candiate k-terms set
			# get new frequent k-terms set
			currFreqTermSet = self.getItemsWithMinSupp(transListSet, currCandiItemSet,itemCountDict, self.minSupp)

		#
		self.itemCountDict = itemCountDict  # 儲存所有候選項以及出現的次數(不僅僅是頻繁項),用來計算置信度
		self.freqSet = freqSet  # dict: freqSet[k] indicate frequent k-term set Lk)
		return itemCountDict, freqSet

	def getSpecRules(self, rhs):
		""" Specify a right item, construct rules for it
		"""
		if rhs not in self.itemSet: # rhs: L1中的某一個item
			print('Please input a term contain in the term-set !')
			return None

		rules = dict()
		for key, value in self.freqSet.items(): #從所有的frequent items set中尋找
			for itemSet in value:
				if rhs.issubset(itemSet) and len(itemSet) > 1: #假設rhs是屬於frequent itemset 中的子集，且該itemSet長度大於1
					item_supp = self.getSupport(itemSet) #取得該itemSet的support值
					itemSet = itemSet.difference(rhs) #取得該itemSet中的parent itemSet
					conf = item_supp / self.getSupport(itemSet) # cond = sup(itemSet) / sup(parent itemSet)
					if conf >= self.minConf:
						rules[itemSet] = conf # 符合最小confidence值集加入規則
		return rules

	def getSupport(self, item):
		""" Get the support of item """
		# return self.itemCountDict[item] / self.transLength
		return self.itemCountDict[item]

	def getJoinedItemSet(self, termSet, k):
		""" Generate new k-terms candiate itemset"""
		return set([term1.union(term2) for term1 in termSet for term2 in termSet
					if len(term1.union(term2)) == k])
		# 取Lk-1中的集合元素，兩兩互配形成Ck

	def getOneItemSet(self, transListSet):
		itemSet = set()
		for line in transListSet:
			for item in line:
				itemSet.add(frozenset([item]))  # 一一取出並納入集合中
		return itemSet

	def getItemsWithMinSupp(self, transListSet, itemSet, freqSet, minSupp):
		""" Get frequent item set using min support
		"""
		itemSet_ = set()
		localSet_ = defaultdict(int)
		for item in itemSet:
			# 統計items在每筆transaction的出現次數(C1,...Cn)
			freqSet[item] += sum([1 for trans in transListSet if item.issubset(trans)])  # 納入frequent itemset集合
			localSet_[item] += sum([1 for trans in transListSet if item.issubset(trans)])  # 納入local frequent itemset集合

		# Only conserve frequent item-set
		n = len(transListSet)  # 取得Transaction Set長度以計算support
		for item, cnt in localSet_.items():
			# itemSet_.add(item) if float(cnt) / n >= minSupp else None
			itemSet_.add(item) if float(cnt) >= minSupp else None  # 統計符合minSupp的frequent items(L1,...Ln)

		return itemSet_


if __name__ == '__main__':

	def loadFromKaggle():

		bakery_data = pd.read_csv('BreadBasket_DMS.csv', encoding='utf-8')
		bakery_data['Date Time'] = bakery_data['Date'] + " " + bakery_data['Time']
		bakery_data = bakery_data.drop(['Date', 'Time'], axis=1)
		bakery_data = bakery_data.drop(['Date Time'], axis=1)
		bakery_data = bakery_data[~bakery_data['Item'].str.contains('NONE')]

		tdl = []
		for i in range(1, bakery_data.Transaction.count() + 1):
			tdf = bakery_data[bakery_data.Transaction == i]
			l = set()
			for j in range(0, tdf.Transaction.count()):
				l.add(tdf.Item.iloc[j])
			if len(l) > 0:
				tdl.append(list(l))
			else:
				tdl.append(None)

		col = ['items']
		TDB = pd.DataFrame({"items": tdl}, columns=col)
		TDB = TDB.dropna()
		return TDB['items'].tolist()


	def loadFromIbm():
		# read data from Ibm
		with open('data', encoding='utf-8') as f:
			content = f.readlines()

		content = [x.strip() for x in content]

		Transaction = []  # to store transaction
		Frequent_items_value = {}  # to store all frequent item sets

		# to fill values in transaction from txt file
		Transaction = {}
		Transaction
		for i in range(0, len(content)):
			rowd = content[i].split(' ')
			rowd = [r for r in rowd if r != '']
			rowd = rowd[1:]
			if Transaction.get(rowd[0], None) == None:
				Transaction[rowd[0]] = [rowd[1]]
			else:
				Transaction[rowd[0]].append(rowd[1])
		# print(type(list(Transaction.values())))
		return list(Transaction.values())


	# cd D:\WorkSpace\PythonWorkSpace\Apriori
	# D:
	# python Apriori.py ibm 2 0.6

	fn = str(sys.argv[1])  # BreadBasket_DMS.csv
	minSup = int(sys.argv[2])  # 3
	minConf = float(sys.argv[3])  # 0.6
	print('fileName = {} ,minSup= {} , minConf={}'.format(fn, minSup, minConf))

	starttime = time.time()
	if fn == 'kaggle':
		dataSet = loadFromKaggle()
	elif fn == 'ibm':
		dataSet = loadFromIbm()

	# Run
	objApriori = Apriori(minSup, minConf)
	itemCountDict, freqSet = objApriori.fit(dataSet)
	#     print(itemCountDict)
	endtime = time.time()
	print("\nTime Taken is: {0:.2f}ms \n".format((endtime - starttime)))
	#   print
	for key, value in freqSet.items():
		print('{}-Itemsets:{}'.format(key, len(value)))
		print('-' * 20)
		for itemset in value:
			print("Items :{} , Support:{} ".format(list(itemset), itemCountDict[itemset]))
		print()

	# Return rules with regard of `rhs`
	print()
	print('List All Rules:')
	print()
	L1 = set(freqSet[1])
	for rhs in L1:
		rules = objApriori.getSpecRules(rhs)
		if len(rules) > 0:
			#             print('-'*20)
			#             print('rules refer to {}'.format(list(rhs)))
			for key, value in rules.items():
				print('Rule : {} -> {}, confidence = {}'.format(list(key), list(rhs), value))
