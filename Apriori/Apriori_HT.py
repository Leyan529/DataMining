import itertools
import time
import sys
import copy

# function to get frequent one itemset
def frequent_one_item(Transaction, min_support):
	candidate1 = {}

	for i in range(0, len(Transaction)):  # 計數C1 item中的出現次數
		for j in range(0, len(Transaction[i])):
			if Transaction[i][j] not in candidate1:
				candidate1[Transaction[i][j]] = 1
			else:
				candidate1[Transaction[i][j]] += 1

	frequentitem1 = []  # to get frequent 1 itemsets with minimum support count
	for value in candidate1:
		if candidate1[value] >= min_support:
			frequentitem1 = frequentitem1 + [[value]]
			Frequent_items_value[tuple(value)] = candidate1[value]

	return frequentitem1


def printL1(L1):
	print("1-itemsets :{}".format(len(L1.items())))
	for key, value in L1.items():
		key = str(key).replace('(', '[').replace(')', ']')
		print("Items: {}, Support :{}".format(key, value))


def print_L(L_value):
	for i, L in enumerate(L_value):
		if i < 2:
			pass
		elif len(L) == 0:
			pass
		else:
			print()
			print("{}-itemsets :{}".format((i), len(L)))
			[print('Items: {} , Support: '.format(itemSet)) for itemSet in L]


# class of Hash node
class Hash_node:
	def __init__(self):
		self.children = {}  # pointer to its children
		self.Is_Leaf = True  # to know the status whether current node is leaf or not
		self.bucket = {}  # contains itemsets in bucket


# class of constructing and getting hashtree
class HashTree:
	# class constructor
	def __init__(self, max_leaf_count, max_child_count):
		self.root = Hash_node()
		self.max_leaf_count = max_leaf_count
		self.max_child_count = max_child_count
		self.frequent_itemsets = []

	# function to recursive insertion to make hashtree
	def recursively_insert(self, node, itemset, index, count):
		if index == len(itemset):  # 如果當前插入索引已滿
			if itemset in node.bucket:  # bucket : HashTree的緩衝儲存區
				node.bucket[itemset] += count  # 若ck itemset已存在bucket中，則在bucket中計數值加一次count
			else:
				node.bucket[itemset] = count  # 若ck itemset已存在bucket中，則在bucket中計數值等於當前count
			return

		if node.Is_Leaf:  # if node is leaf
			if itemset in node.bucket:
				node.bucket[itemset] += count  # 若ck itemset已存在bucket中，則在bucket中計數值加一次count
			else:
				node.bucket[itemset] = count  # 若ck itemset已存在bucket中，則在bucket中計數值等於當前count
			if len(node.bucket) == self.max_leaf_count:  # 如果bucket可容納的數量已經達到達到leaf node的乘載最大數量
				for old_itemset, old_count in node.bucket.items():  # 取出bucket中舊有的itemset和count

					hash_key = self.hash_function(old_itemset[index])  # 將其 hashing 至另外一個 index
					if hash_key not in node.children:  # 如果Node底下並沒有以hash_key為首的children HT
						node.children[hash_key] = Hash_node()  # 在Node底下做一棵以hash_key為首的children HT
					self.recursively_insert(node.children[hash_key], old_itemset, index + 1, old_count)
				# 將bucket中舊有的itemset遞迴插入至children HT
				# since no more requirement of this bucket
				del node.bucket  # 將緩衝儲存區的內容清除
				node.Is_Leaf = False  # 該節點成為Non-Leaf node
		else:  # if node is not leaf
			hash_key = self.hash_function(itemset[index])
			if hash_key not in node.children:  # 如果Non-Leaf node底下並沒有以hash_key為首的children HT
				node.children[hash_key] = Hash_node()  # 在Node底下做一棵以hash_key為首的children HT
			self.recursively_insert(node.children[hash_key], itemset, index + 1, count)
		# 將itemset遞迴插入至指定的Hash children HT

	def insert(self, itemset):
		itemset = tuple(itemset)  # 將ck中的itemset轉化成dict結構
		self.recursively_insert(self.root, itemset, 0, 0)  # 進行遞迴插入

	# to add support to candidate itemsets. Transverse the Tree and find the bucket in which this itemset is present.
	def add_support(self, itemset):
		Transverse_HNode = self.root
		itemset = tuple(itemset)
		index = 0
		while True:
			if Transverse_HNode.Is_Leaf:
				if itemset in Transverse_HNode.bucket:  # found the itemset in this bucket
					Transverse_HNode.bucket[itemset] += 1  # increment the count of this itemset.
				break
			hash_key = self.hash_function(itemset[index])
			if hash_key in Transverse_HNode.children:
				Transverse_HNode = Transverse_HNode.children[hash_key]
			else:
				break
			index += 1

	# to transverse the hashtree to get frequent itemsets with minimum support count
	def get_frequent_itemsets(self, node, support_count, frequent_itemsets):
		if node.Is_Leaf:
			for key, value in node.bucket.items():
				if value >= support_count:  # if it satisfies the condition
					frequent_itemsets.append(list(key))  # then add it to frequent itemsets.
					Frequent_items_value[key] = value
			return

		for child in node.children.values():
			self.get_frequent_itemsets(child, support_count, frequent_itemsets)

	# hash function for making HashTree
	def hash_function(self, val):
		return int(val) % self.max_child_count


# To generate hash tree from candidate itemsets
def generate_hash_tree(candidate_itemsets, max_leaf_count, max_child_count):
	htree = HashTree(max_child_count, max_leaf_count)  # create instance of HashTree
	for itemset in candidate_itemsets:
		htree.insert(itemset)  # to insert itemset into Hashtree
	return htree


# to generate subsets of itemsets of size k
def generate_k_subsets(dataset, length):
	subsets = []
	for itemset in dataset:
		subsets.extend(map(list, itertools.combinations(itemset, length)))
	return subsets


def subset_generation(ck_data, l):
	# itertools.combinations(iterable, r)
	# 創建一個迭代器，返回iterable中所有長度為r的子序列，返回的子序列中的項按輸入iterable中的順序排序（不帶重複）
	return map(list, set(itertools.combinations(ck_data, l)))


# apriori generate function to generate ck
def apriori_generate(Lk, k):
	ck = []
	# join step
	lenlk = len(Lk)
	for i in range(lenlk):  # 從Lk中的itemsets取出元素相互混種
		for j in range(i + 1, lenlk):
			L1 = list(Lk[i])[:k - 2]
			L2 = list(Lk[j])[:k - 2]
			if L1 == L2:
				ck.append(sorted(list(set(Lk[i]) | set(Lk[j]))))  # 將新混種出來的itemsets放入Ck

			# prune step
	final_ck = []
	for candidate in ck:
		all_subsets = list(subset_generation(set(candidate), k - 1))  # 產生Ck-1的Subsets
		found = True

		# -- 改良--#
		for subset in all_subsets:
			subset = list(sorted(subset))
			if (subset not in Lk) and (subset in ck):
				ck.remove(subset)

			# -- 舊有--#
			# for i in range(len(all_subsets)):
			#     value = list(sorted(all_subsets[i]))
			#     if value not in Lk:
			#         found = False
			# if found == True:
			#     final_ck.append(candidate)

	return ck, final_ck


def generateL(ck, min_support):
	support_ck = {}
	for val in Transaction1:
		for val1 in ck:
			value = set(val)
			value1 = set(val1)

			if value1.issubset(value):
				if tuple(val1) not in support_ck:
					support_ck[tuple(val1)] = 1
				else:
					support_ck[tuple(val1)] += 1
	frequent_item = []
	for item_set in support_ck:
		if support_ck[item_set] >= min_support:
			frequent_item.append(sorted(list(item_set)))
			Frequent_items_value[item_set] = support_ck[item_set]

	return frequent_item


# main apriori algorithm function
def apriori(L1, min_support, max_leaf_count=3, max_child_count=5):
	k = 2;
	L = []
	L.append(0)
	L.append(L1)

	start = time.time()
	while (len(L[k - 1]) > 0):  # 假如Lk-1裡面還有itemsets就繼續做
		ck, final_ck = apriori_generate(L[k - 1], k)  # to generate candidate itemsets
		h_tree = generate_hash_tree(ck, max_leaf_count, max_child_count)  # to generate hashtree
		k_subsets = generate_k_subsets(Transaction1, k)  # to generate subsets of each transaction
		for subset in k_subsets:
			h_tree.add_support(subset)  # hashtree中的每個node加上support
		lk = []
		h_tree.get_frequent_itemsets(h_tree.root, min_support, lk)  # 從hashtree中取出每個 frequent itemsets
		L.append(lk)
		k = k + 1  # Ck,Lk提升一階
	end = time.time()
	return L, (end - start)


def generateL1(L1, Transaction_len):
	print("1-itemsets :{}".format(len(L1.items())))
	cpL1 = copy.deepcopy(L1)

	for key, value in cpL1.items():
		# key = str(key).replace('(', '[').replace(')', ']')
		# key = int(key)
		item = 0
		for i in range(0,len(key)):
			item = item + int(key[i])
			item*=10
			L1[item] = value
		del L1[key]
		print("Items: {} , Support :{}".format([item], value))


def generateLn(L_value):
	for i, L in enumerate(L_value):
		if i < 2:
			pass
		elif len(L) == 0:
			pass
		else:
			print()
			print("{}-itemsets :{}".format((i), len(L)))
			[print('Items: {} , Support:{}'.format(itemSet,Frequent_items_value[tuple(itemSet)])) for itemSet in L]


if __name__ == '__main__':
	import itertools
	import time


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
	# python Apriori_HT.py ibm 10 3 5
	fn = str(sys.argv[1])
	minSup = int(sys.argv[2])  # 3
	max_leaf_count = int(sys.argv[3])  # 3
	max_child_count = int(sys.argv[4])  # 5
	print('fileName = {} ,minSup= {} , max_leaf_count={} , max_child_count={} '.format(fn, minSup, max_leaf_count,max_child_count))

	starttime = time.time()

	Frequent_items_value = {}  # to store all frequent item sets

	if fn == 'ibm':
		Transaction = loadFromIbm()

	Transaction_len = len(Transaction)

	print("All frequent itemsets with their support count:")
	L1 = frequent_one_item(Transaction, minSup)
	generateL1(Frequent_items_value, Transaction_len)

	# to remove infrequent 1 itemsets from transaction
	Transaction1 = []
	for i in range(0, len(Transaction)):
		list_val = []
		for j in range(0, len(Transaction[i])):
			if [Transaction[i][j]] in L1:
				list_val.append(Transaction[i][j])
		Transaction1.append(list_val)

	L_value, time_taken = apriori(L1, minSup, max_leaf_count, max_child_count)
	generateLn(L_value)

	print("\nTime Taken is: {0:.3f} ms\n".format(time_taken))
