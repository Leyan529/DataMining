#
import copy
import time
import sys
import pandas as pd
import numpy as np
import re


class Item:
	elements = []
	supp = 0

	def __init__(self, elements, supp=0):
		self.elements = elements
		self.supp = supp

	def __str__(self):
		returnstr = '[ '
		for e in self.elements:
			returnstr += e + ','
		returnstr += ' ]' + ' (support :%d)\t' % (self.supp)
		return returnstr

	def getSubset(self, k, size):
		subset = []
		if k == 1:
			for i in range(size):
				subset.append([self.elements[i]])
			return subset
		else:
			i = size - 1
			while i >= k - 1:
				myset = self.getSubset(k - 1, i)
				j = 0
				while j < len(myset):
					# Attention a+=b  a=a+b
					myset[j] += [self.elements[i]]  # Why Elements change here?
					j += 1
				subset += (myset)
				i -= 1
			return subset

	def lastDiff(self, items):
		length = len(self.elements)
		if length != len(items.elements):  # length should be the same
			return False  # 兩邊長度要一樣才能進行混種
		if self.elements == items:  # if all the same,return false
			return False  # 兩邊內容若完全一樣不行進行混種
		return self.elements[0:length - 1] == items.elements[0:length - 1]

	# 兩邊內容最後一個元素以外的元素必須完全相同才能進行混種

	def setSupport(self, supp):
		self.supp = supp

	def join(self, items):
		temp = copy.copy(self.elements)
		temp.insert(len(self.elements), items.elements[len(items.elements) - 1])
		it = Item(temp, 0.0)
		return it


class C:
	'''candidate '''
	elements = []
	k = 0  # order

	def __init__(self, elements, k):
		self.elements = elements
		self.k = k

	def isEmpty(self):
		if len(self.elements) == 0:
			return True
		return False

	# get the same order of itemsets whose support is at lease the threshold
	def getL(self, threshold):  # 計算L2~Lk的function
		items = []
		for item in self.elements:
			if item.supp >= threshold:
				items.append(copy.copy(item))
		if len(items) == 0:
			return L([], self.k)
		return L(copy.deepcopy(items), self.k)

	def __str__(self):
		returnstr = str(self.k) + '-itemset:' + str(len(self.elements)) + ' \r\n{ '
		for e in self.elements:
			if True == isinstance(e, Item):
				returnstr += e.__str__()
		returnstr += ' }'
		return returnstr


class L:
	'''store all the  1-itemsets,2-itemsets,...k-itemsets'''
	items = []  # all the item in order K
	k = 0

	def __init__(self, items, k):
		self.items = items
		self.k = k

	def has_inFrequentItemsets(self, item):
		#        return False
		#        #先不優化
		subs = item.getSubset(self.k, len(item.elements))  # 取出候選item的subsets集合 ex:{ABC} ->{AB,AC,BC}
		for each in subs:  # 依序取出每個subset
			flag = False  # 代表目前subset是否為FrequentItemset
			for i in self.items:  # 依序從Lk中取出每個items (Lk中都是FrequentItemset)
				if i.elements == each:
					flag = True  # 是，則直接判斷下一個subset
					break
			if flag == False:
				return True  # 發現有subset為inFrequentItemset，則該候選item為inFrequentItemset

		return False  # 全部的subset皆為FrequentItemset，候選item不在inFrequentItemset中

	def aprioriGen(self):  # Generate Ck
		length = len(self.items)
		result = []  # store Ck
		for i in range(length):
			for j in range(i + 1, length):
				if self.items[i].lastDiff(self.items[j]):  # 若符合混種條件
					item = self.items[i].join(self.items[j])  # 將Lk中的items交配混種，得k階候選item
					if False == self.has_inFrequentItemsets(item):
						# 用Apriori性質：任一頻繁項集的所有非空子集也必須是頻繁的，
						# 反之，如果某個候選的非空子集不是頻繁的，那麼該候選肯定不是頻繁的，從而可以將其從CK中刪除。
						result.append(item)
		if (len(result) == 0):
			return C([], self.k + 1)
		return C(result, self.k + 1)  # 回傳提升一階後的k & Ck

	def __str__(self):
		returnstr = "\r\n" + str(self.k) + '-itemsets :' + str(len(self.items)) + "\r\n{"
		for item in self.items:
			returnstr += item.__str__()
		returnstr += '}'
		return returnstr


class LS:
	'''store from L1-itemset to Lk-itemset'''
	values = {}  # L1,L2,Lk

	def get(self, k):
		return self.values[k]

	def size(self):
		return len(self.values)

	def put(self, l, k):
		self.values[k] = l

	def isEmpty(self):
		return self.size() == 0

	def __str__(self):
		returnstr = '-----result--------\r\n'
		for l in self.values:
			returnstr += self.values[l].__str__()
		return returnstr


class Rule:
	confidence = .0
	str_rule = ''

	def __init__(self, confidence, str_rule):
		self.confidence = confidence
		self.str_rule = str_rule

	def __str__(self):
		return 'Rule:' + self.str_rule + '  confidence:' + str(self.confidence)


class Apriori:
	def __init__(self, data, min_supp=2):
		self.data = []
		self.size = 0
		self.min_supp = min_supp
		self.data = data
		self.size = len(self.data)

	def findFrequent1Itemsets(self):
		totalItemsets = []  # store all the item from transaction
		for temp in self.data:
			totalItemsets.extend(temp)
		items = []  # store the 1-itemset s

		while len(totalItemsets) > 0:
			item = totalItemsets[0]  # get unique item from itemset
			count = 0
			j = 0
			while j < len(totalItemsets):  # calc item count
				if (item == totalItemsets[j]):
					count += 1
					totalItemsets.remove(item)  # remove the first occurence
				else:
					j += 1
				#             t_supp = count / self.size  # calc unique item support
			t_supp = count  # calc unique item support

			if np.float64(t_supp) >= np.float64(self.min_supp):
				items.append(Item([item], t_supp))  # 1-itemset ([item1,...,item_n],support)

		temp = L(copy.deepcopy(items), 1)  # show info from 1-itemset
		return temp

	def ralationRules(self, maxSequence, min_confidence):
		# maxSequence : Lk中的所有itemsets
		ruls = []  # 存放所有的關聯式規則
		for each in maxSequence:
			for i in range(len(each.elements) - 1):  # real subsets
				subsets = each.getSubset(i + 1, len(each.elements))  # 對於每一個itemset計算出 (2^k)-2 個subset ,ex:k=4 14個subset
				for subset in subsets:  # 從subsets集合中取出每一個subset
					count = 0
					for tran_item in self.data:
						flag = False  # 標記subset中的每個元素都在源中出現
						for ele in subset:  # 從每一個subset中取出元素ele判斷是否在來源transaction中出現過
							if ele not in tran_item:
								flag = True
								break
						if flag == False:
							count += 1  # subset出現在原始資料集中，計數值+1
					confidence = each.supp / count  # 計算當前Lk itemset的信心值
					# confidence = (Lk itemset出現次數) / (subset出現次數)
					if confidence >= min_confidence:  # confidence/the number of the frequent pattern
						# 計算由Lk itemSet 所衍伸出來的關聯式規則
						# set(each.elements) - set(subset) Lk itemset與subset的差集 (代表衍伸會購買的物品)
						str_rule = str(set(subset)) + '-->' + str(set(each.elements) - set(subset))
						rule = Rule(confidence, str_rule)
						ruls.append(rule)  # 加入一條新的關聯規則
		return ruls

	def do(self):
		ls = LS()
		oneitemset = self.findFrequent1Itemsets()
		ls.put(oneitemset, 1)  # ls add L1 itemset
		k = 2
		while False == ls.isEmpty():  # if Lk itemset is not empty : do-loop
			cand = ls.get(k - 1).aprioriGen()  # pick L(k-1) to generate Ck
			if cand.isEmpty():
				break  # if Ck is empty : break
			for each in cand.elements:  # 針對Ck裡每個item，從原來的itemset中找出符合規則的item
				count = 0
				for each_src in self.data:  # 判斷原先data set中的關聯
					if len(each_src) < len(each.elements):  # pass 掉原先資料長度不夠的itemset
						pass
					else:
						# 不是必須連續 相等才滿足條件，只要元素都在裡面即可
						flag = True
						for just_one_e in each.elements:  # 針對Ck裡每個item，檢查是否存在於原先data set中
							flag = just_one_e in each_src
							if flag == False:  # 只要有一個不在，即退出
								break
						if flag == True:  # 當前候選事件都在的話，計數
							count += 1

				supp = count  # 計算當前候選itemset的support
				each.setSupport(supp)
			ls.put(cand.getL(apriori.min_supp), k)  # 以當前候選itemset為參數，計算Lk，並放入Lk set
			k += 1
		return ls


def getFinalRule(apriori, ls, minConf):  # 列印最後規則
	final_Lk = ls.get(ls.size())  # 取出最後的Lk
	print(final_Lk)
	rules = apriori.ralationRules(final_Lk.items, min_confidence=minConf)
	for rule in rules:
		print(rule)


def getAllRule(apriori, ls, minConf):  # 列印全部規則
	for i in range(1, ls.size()):
		Lk = ls.get(i)
		rules = apriori.ralationRules(Lk.items, min_confidence=minConf)
		for rule in rules:
			print(rule)


def getPatterns(ls):
	new_ls = {}
	for i in range(2, ls.size() + 1):
		fls = str(ls.get(i)).split('\t')
		tt_fls = fls[0].split('\r\n')
		title = tt_fls[1]
		tt_fls[2] = tt_fls[2].replace('{', '')

		new_fl = []
		new_fl.extend([tt_fls[2]])
		new_fl.extend(fls[1:])
		new_fl = [l for l in new_fl if l != '}']
		new_ls[title] = new_fl

	for title, l in new_ls.items():
		print(title)
		[print("Items:{}".format(items)) for items in l]
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

	apriori = Apriori(dataSet, min_supp=minSup)
	ls = apriori.do()
	endtime = time.time()
	print("\nTime Taken is: {}\n".format((endtime - starttime) * 1000))
	print()
	getPatterns(ls)
	print('List All Rules')
	# getFinalRule(apriori,ls,minConf)
	getAllRule(apriori, ls, minConf)
