import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import sys


def transfer2FrozenDataSet(dataSet):
	frozenDataSet = {}
	for elem in dataSet:
		frozenDataSet[frozenset(elem)] = 1
	return frozenDataSet


class TreeNode:
	def __init__(self, nodeName, count, nodeParent):
		self.nodeName = nodeName
		self.count = count
		self.nodeParent = nodeParent
		self.nextSimilarItem = None  # 指向下一個相同元素的指針nextSimilarItem
		self.children = {}

	def updateC(self, count):
		self.count += count


def createFPTree(frozenDataSet, minSupport):
	# scan dataset at the first time, filter out items which are less than minSupport
	frequenctNodeTable = {}  # 紀錄每個item的出現數目,還有其parent Node
	for transactions in frozenDataSet:  # 依序取出每筆transactions
		for item in transactions:  # 在取出每項item
			item = str(item)
			frequenctNodeTable[item] = frequenctNodeTable.get(item, 0) + frozenDataSet[transactions]  # 取出每項的count+1


			# count = frequenctNodeTable.get(item, default = 0) , 1 = frozenDataSet[transactions]
	frequenctNodeTable = {k: v for k, v in frequenctNodeTable.items() if v >= minSupport}  # 濾除掉不滿足minSupport的item
	frequentItemSet = set(frequenctNodeTable.keys())  # 紀錄滿足minSupport items的集合

	if len(frequentItemSet) == 0: return None, None

	for k in frequenctNodeTable:
		# 將frequenctNodeTable中的結構(item,count) -> (item,[count,ItemNode(default=None)])
		frequenctNodeTable[k] = [frequenctNodeTable[k], None]  # (初始化所有非root Node的所有資訊)

	fptree = TreeNode("null", 1, None)  # 初始化FP-Tree Root Node
	# scan dataset at the second time, filter out items for each record
	for transactions, count in frozenDataSet.items():  # 再次scanfrozenDataSet中的所有items
		frequentItemsInRecord = {}  # 紀錄transaction中的frequent items
		for item in transactions:
			if item in frequentItemSet:  # 利用frequentItemSet過濾出每筆transaction中的常見items
				frequentItemsInRecord[item] = frequenctNodeTable[item][0]
		if len(frequentItemsInRecord) > 0:
			# 將transaction中的items依照support大小作排序(大的在前)，形成一條FP path
			orderedFrequentItems = [v[0] for v in sorted(frequentItemsInRecord.items(), key=lambda v: v[1],
														 reverse=True)]  # sort by v[1] (support)降序
			updateFPTree(fptree, orderedFrequentItems, frequenctNodeTable, count)

	return fptree, frequenctNodeTable


def updateFPTree(fptree, orderedFrequentItems, frequenctNodeTable, count):
	# handle the first item
	firstOrderItem = orderedFrequentItems[0]  # scan FP-path 的第1個item
	if firstOrderItem in fptree.children:  # 如果firstOrderItem是fptree 節點的子節點
		fptree.children[firstOrderItem].updateC(count)  # 增加該子節點的count數
	else:
		fptree.children[firstOrderItem] = TreeNode(firstOrderItem, count,
												   fptree)  # 否則，需要新創建一個TreeNode 節點，然後將其賦給fptree 節點的子節點
		# update frequenctNodeTable
		if frequenctNodeTable[firstOrderItem][1] == None:  # 如果frequenctNodeTable裡記錄的firstOrderItem Node為None
			frequenctNodeTable[firstOrderItem][1] = fptree.children[
				firstOrderItem]  # 則更新frequenctNodeTable裡的firstOrderItem Node為自己本身
		else:
			updateFrequenctNodeTable(frequenctNodeTable[firstOrderItem][1],
									 fptree.children[firstOrderItem])  # 更新 firstOrderItem 下一個相似元素指標
	# handle other items except the first item
	if (len(orderedFrequentItems) > 1):  # 假如 FP-path 的後面還有其他item
		updateFPTree(fptree.children[firstOrderItem], orderedFrequentItems[1:], frequenctNodeTable, count)
		# 以FP-path上的firstOrderItem node為新的root Node，處理後續的orderedFrequentItems


def updateFrequenctNodeTable(headFrequentNode, targetNode):
	while (headFrequentNode.nextSimilarItem != None):
		headFrequentNode = headFrequentNode.nextSimilarItem  # 不斷尋找相似元素指標為空的headFrequentNode
	headFrequentNode.nextSimilarItem = targetNode  # 將headFrequentNode 與 targetNode 以相似元素指標連接


def mining_FPTree(frequenctNodeTable, suffix, frequentPatterns, minSupport):
	# for each item in frequenctNodeTable, find conditional suffix path, create conditional fptree, then iterate until there is only one element in conditional fptree
	frequenctItems = [v[0] for v in sorted(frequenctNodeTable.items(),
										   key=lambda v: v[1][0])]  # 將frequenctNodeTable以升序排序，從low frequent開始mining
	if (len(frequenctItems) == 0): return

	for frequenctItem in frequenctItems:
		newSuffix = suffix.copy()  # 只拷貝父級目錄，子目錄不拷貝
		newSuffix.add(frequenctItem)  # 加入一個擁有最低出現次數的Suffix item作為newSuffix
		support = frequenctNodeTable[frequenctItem][0]  # 逐一取得Suffix item的support
		frequentPatterns[frozenset(newSuffix)] = support  # 逐一取得Suffix item的support，並將其一一加入到frequentPatterns
		print(frequentPatterns)

		suffixPath = getSuffixPath(frequenctNodeTable, frequenctItem)  # Find suffix patterns to the frequenctItem
		if (suffixPath != {}):
			conditionalFPtree, conditionalFNodeTable = createFPTree(suffixPath, minSupport)
			# 根據suffixPath建立conditionalFPtree 和 conditional frequenctNode Table
			if conditionalFNodeTable != None:  # 假如 conditional frequenctNode不為空
				mining_FPTree(conditionalFNodeTable, newSuffix, frequentPatterns,
							  minSupport)  # 繼續挖掘剩餘的conditionalFPtree


def getSuffixPath(frequenctNodeTable, frequenctItem):
	suffixPath = {}
	beginNode = frequenctNodeTable[frequenctItem][1]  # 指向當前frequenctItem 的 Tree Node
	suffixs = ascendNodeList(beginNode)  # suffixs紀錄由beginNode往上層尋找的ascendTree (不包含beginNode)
	if ((suffixs != [])):
		suffixPath[
			frozenset(suffixs)] = beginNode.count  # 建立一條由suffix Node起始但不包含suffix Node的完整suffixPath，數量為beginNode的出現次數

	while (beginNode.nextSimilarItem != None):
		beginNode = beginNode.nextSimilarItem
		suffixs = ascendNodeList(beginNode)
		if (suffixs != []):
			suffixPath[frozenset(suffixs)] = beginNode.count

	return suffixPath  # 回傳suffixPath


def ascendNodeList(treeNode):
	suffixs = []
	while ((treeNode.nodeParent != None) and (treeNode.nodeParent.nodeName != 'null')):
		treeNode = treeNode.nodeParent
		suffixs.append(treeNode.nodeName)
	return suffixs


def rulesGenerator(frequentPatterns, minConf, rules):
	for frequentset in frequentPatterns:
		if (len(frequentset) > 1):  # 依序取得所有長度大於1的frequentset
			getRules(frequentset, frequentset, rules, frequentPatterns, minConf)
			# 透過frequentset自己本身去產生規則並輸出到rules內


def removeStr(set, str):
	tempSet = []
	for elem in set:
		if (elem != str):
			tempSet.append(elem)  # 將不等於frequentElem的項目篩選出來，並加入到tempSet中
	tempFrozenSet = frozenset(tempSet)
	return tempFrozenSet


def getRules(frequentset, currentset, rules, frequentPatterns, minConf):  # 由大至小拆解各集合，以取得規則
	for frequentElem in currentset:
		subSet = removeStr(currentset, frequentElem)  # subSet為不包含frequentElem的子集合
		# print(currentset)
		# print(frequentElem)
		confidence = frequentPatterns.get(frequentset) / frequentPatterns.get(subSet, 9999)
		# confidence = sup(currentset) / sup(subSet)
		if (confidence >= minConf):
			flag = False
			for rule in rules:
				# rule[0] : 推演規則  # rule[1] : 衍伸規則
				if (rule[0] == subSet and rule[1] == frequentset - subSet):
					flag = True
			if (flag == False):
				rules.append((subSet, frequentset - subSet, confidence))
				# rules.append (推演規則 -------> 衍伸規則 , confidence)

			if (len(subSet) >= 2):  # 如果subSet中的frequent items數大於2
				getRules(frequentset, subSet, rules, frequentPatterns, minConf)
				# 以subSet為目標繼續挖掘其他的規則


def getpatterns(pattern):
	maxVal = max(pattern.items())[1]
	dictpattern = {}
	for i in range(1, maxVal + 10):
		dictpattern[str(i)] = []

	for key, value in pattern.items():
		dictpattern[str(len(key))].append([key, value])

	for i in range(1, maxVal + 10):
		patterns = dictpattern[str(i)]
		if len(patterns) > 0:
			print("{}-items set".format(str(i)))
			[print("items:{} , support:{} ".format(set(item[0]), item[1])) for item in patterns]
			print()


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
	# python FP-Growth.py ibm 3 0.6
	fn = str(sys.argv[1])  # BreadBasket_DMS.csv
	minSup = int(sys.argv[2])  # 3
	minConf = float(sys.argv[3])  # 0.6
	print('fileName = {} ,minSup= {} , minConf={}'.format(fn, minSup, minConf))

	starttime = time.time()
	if fn == 'kaggle':
		dataSet = loadFromKaggle()
	elif fn == 'ibm':
		dataSet = loadFromIbm()

	frozenDataSet = transfer2FrozenDataSet(dataSet)  # 為transaction中所有曾經出現的item建立Node，並初始化support為1
	minSupport = 3
	fptree, frequenctNodeTable = createFPTree(frozenDataSet, minSup)
	frequentPatterns = {}

	suffix = set([])
	mining_FPTree(frequenctNodeTable, suffix, frequentPatterns, minSup)

	rules = []
	endtime = time.time()
	print("fptree:")
	print("\nTime Taken is: {}\n".format((endtime - starttime) * 1000))

	print("frequent patterns:")
	# getpatterns(frequentPatterns)

	print(frequentPatterns)
	print("association rules:")
	# rulesGenerator(frequentPatterns, minConf, rules)
	# rules = [rule for rule in rules if rule != None]
	# [print('Rules:{}-->{}, confidence:{}'.format(set(r[0]), set(r[1]), r[2])) for r in rules]
