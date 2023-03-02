# China_10_most_common_surnames_by_province
# 

import pandas as pd
import numpy as np
#import pandas.plotting._matplotlib as plt
import matplotlib.pyplot as plt
import networkx as nx


CONST_NUM_PROV = 28

# use Pandas to read in surname distrib. data from csv file
raw = pd.read_csv('China_top10surname.csv', skiprows=1, names=['Province','1','2','3','4','5','6','7','8','9','10'])
data = np.array(raw)

# remove column with province names
data_format = data[:, 1:]

#print(raw)
#print(data_format)
print()

'''
Clustering of provinces by surname distribution- 
For each "province", calculate similarity of surname frequency distribution with all other "provinces". 

Similarity metric- 
1) 1 point for each surname (number) found in both the "test" province and the "reference" province (max possible: 10)
2) 1 point for each surname (number) found in the same position in both provinces for indices 4 to 10 (max possible: 7)
3) 1 point for each surname (number) found in the same position in both provinces for indices 2 to 3 (max posssible: 2)
4) 2 points for matching most common surnames (max possible: 2)
'''

# Python function for similarity metric 1
# compare contents of 2 arrays assuming they have the same length
def compare_sets_shared(a, b):
  result = set(a).intersection(set(b))
  return len(result)

# Python function for similarity metric 2
# comparing the elements of 2 arrays from indices 3 to 9, assuming both have a length of 10
def identical_4to10(a, b):
  if (len(a) != len(b) or len(a) != 10):
    return -1
  tally = 0
  for i in range(3, 10):
    if a[i] == b[i]:
      tally += 1
  return tally

# Python function for similarity metric 3
# comparing the elements of 2 arrays from indices 1 to 2, assuming both have a length of 10
def identical_2and3(a, b):
  if (len(a) != len(b) or len(a) != 10):
    return -1
  tally = 0
  for i in range(1, 2):
    if a[i] == b[i]:
      tally += 1
  return tally


# Python function for similarity metric 4
# comparing the elements of 2 arrays from indices 1 to 2, assuming both have a length of 10
def identical_1(a, b):
  if (len(a) != len(b) or len(a) != 10):
    return -1
  if a[0] == b[0]:
      return 2
  return 0

'''
Create confusion matrix for all 28 provinces
'''

'''
Map visualization (models the 28 provinces as a minimal spanning tree)

https://www.youtube.com/watch?v=LFT0rj-xvYA for Prim's algorithm

Dec 2022 update- https://www.ecosia.org/search?addon=chrome&addonversion=5.1.2&q=how%20to%20store%20chinese%20characters%20as%20information%20in%20python

Idea- eventually edit each element in the map to be a data structure that contains a province (abbreviated, full English, full Chinese), and a surname 
(ideally in converted format and with the original character being displayed with pinyin also displayed)...

'''


# source: https://i.imgur.com/m8M4Dwj.jpg
prc_prov_adj_matrix = [[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Manchuria
                    [0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Shandong
                    [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Beijing
                    [1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Hebei
                    [0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Shanxi
                    [0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # InnerMongolia
                    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Tianjin
                    [0,1,0,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], # Henan
                    [0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0], # Shaanxi
                    [0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # Gansu
                    [0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Ningxia
                    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # Xinjiang
                    [0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0], # Anhui
                    [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0], # Jiangsu
                    [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], # Qinghai
                    [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0], # Hubei
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0], # Chongqing
                    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0], # Sichuan
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1], # Guizhou
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1], # Hunan
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1], # Yunnan
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0], # Jiangxi
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # Shanghai
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0], # Zhejiang
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0], # Fujian
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,1], # Guangdong
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # Hainan
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0]] # Guangxi

prc_prov_adj_matrix_ = np.array(prc_prov_adj_matrix)


# create similarity matrix for all pairs of provinces
prc_prov_similarity_matrix = np.zeros((CONST_NUM_PROV, CONST_NUM_PROV))

# loop through the various combinations of rows in the dataset table
for i in range(CONST_NUM_PROV):
    #prc_prov_similarity_matrix[i, j] = 25 # highest possible value in my ranking
    for j in range(i+1, CONST_NUM_PROV):
        val1 = compare_sets_shared(data_format[i], data_format[j])
        val2 = identical_4to10(data_format[i], data_format[j])
        val3 = identical_2and3(data_format[i], data_format[j])
        val4 = identical_1(data_format[i], data_format[j])
        # error handling hypothetical
        if (val1 == -1 or val2 == -1 or val3 == -1 or val4 == -1):
            prc_prov_similarity_matrix[i, j] = 0
            prc_prov_similarity_matrix[j, i] = 0
        else:
            prc_prov_similarity_matrix[i, j] = val1 + val2 + val3 + val4
            prc_prov_similarity_matrix[j, i] = prc_prov_similarity_matrix[i, j]

#print(prc_prov_similarity_matrix)

'''
prc_prov_sim_matrix_reload = prc_prov_similarity_matrix
for a in range(CONST_NUM_PROV):
    for b in range(CONST_NUM_PROV):
        if prc_prov_adj_matrix_[a, b] == 0:
            prc_prov_sim_matrix_reload[a, b] = 0

print(prc_prov_sim_matrix_reload)
'''


# https://www.linkedin.com/pulse/beginners-guide-choropleth-map-python-visualize-chinas-ying-li/
# for visualization of China on an actual map?



# general thought process for creating the graph from the existing adjacency matrices
# https://stackoverflow.com/questions/42558165/load-nodes-with-attributes-and-edges-from-dataframe-to-networkx
# https://www.youtube.com/watch?v=TlkpoB3JAHE

# resources for handling Chinese characters and setting up data frames for graphsh
# https://stackoverflow.com/questions/42558165/load-nodes-with-attributes-and-edges-from-dataframe-to-networkx
# https://stackoverflow.com/questions/51333489/chinese-encoding-in-python

'''
1. Create pandas dataframe of each node's name in English abbreviation, full English name, and Chinese name- nodes are in same order as in matrices
2. Create and add each node from the dataframe so that all 28 nodes are in the graph
3. Make a duplicate of this set of nodes for another graph
4. Part I- iteratively create edges for all pairs of nodes that are connected to each other per prc_prov_adj_matrix, with weights taken from prc_prov_sim_matrix_reload.
5. Part II- iteratively create edges for all pairs of nodes that have values from prc_prov_sim_matrix >= 10
6. Display graphs 1 and 2? 
'''

G0 = nx.Graph()

G0.graph["Name"] = "PRC provinces surnames"

#WIP- idea is to have the nodes numbered from 0 to 27 (indices of the adj matrix), while the province abbreviation will be displayed in the node
G0.add_nodes_from([
    ("0", {"EN": "MAN", "Zh-S": "东北"}),
    ("1", {"EN": "SD", "Zh-S": "山东"}),
    ("2", {"EN": "BJ", "Zh-S": ""}),
    ("3", {"EN": "HEB", "Zh-S": "河北"}),
    ("4", {"EN": "SW", "Zh-S": "山西"}),
    ("5", {"EN": "MON", "Zh-S": "内蒙"}),
    ("6", {"EN": "TJ", "Zh-S": "天津"}),
    ("7", {"EN": "HEN", "Zh-S": "河南"}),
    ("8", {"EN": "SX", "Zh-S": "陕西"}),
    ("9", {"EN": "GS", "Zh-S": "甘肃"}),
    ("10", {"EN": "NX", "Zh-S": "宁夏"}),
    ("11", {"EN": "XJ", "Zh-S": "新疆"}),
    ("12", {"EN": "AH", "Zh-S": "安徽"}),
    ("13", {"EN": "JS", "Zh-S": "江苏"}),
    ("14", {"EN": "QH", "Zh-S": "青海"}),
    ("15", {"EN": "HUB", "Zh-S": "湖北"}),
    ("16", {"EN": "CQ", "Zh-S": "重庆"}),
    ("17", {"EN": "SC", "Zh-S": "四川"}),
    ("18", {"EN": "GZ", "Zh-S": "贵州"}),
    ("19", {"EN": "HUN", "Zh-S": "湖南"}),
    ("20", {"EN": "YUN", "Zh-S": "云南"}),
    ("21", {"EN": "JX", "Zh-S": "江西"}),
    ("22", {"EN": "SH", "Zh-S": "上海"}),
    ("23", {"EN": "ZJ", "Zh-S": "浙江"}),
    ("24", {"EN": "FJ", "Zh-S": "福建"}),
    ("25", {"EN": "GD", "Zh-S": "广东"}),
    ("26", {"EN": "HAN", "Zh-S": "海南"}),
    ("27", {"EN": "GX", "Zh-S": "广西"})
])

# make copy of existing graph's nodes
G1 = G0.copy()
# make copy of copied graph's nodes
G2 = G1.copy()

# create copy of the similarity matrix with all values under 10 truncated
# goal is to visualize regional clusters by limiting the number of edges with weight >= 10 per node
sim_matrix_trunc = np.matrix.copy(prc_prov_similarity_matrix)

# https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix/43171216#43171216
# for each row in the resultant matrix, zero out all values below the 3rd largest value

# loop through the first 13 nodes to handle the northern provinces + Anhui
temp0 = np.copy(sim_matrix_trunc[0:13])
#print(temp0)
for i in range(13):
    tmp1 = temp0[i]
    val = np.partition(tmp1.flatten(), -3)[-3]
    tmp1[tmp1 < val] = 0
    for j in range(CONST_NUM_PROV):
        sim_matrix_trunc[i, j] = tmp1[j]
        #sim_matrix_trunc[j, i] = sim_matrix_trunc[i, j]

# also repeat this step for node 14 (Qinghai)
QH_edges = np.copy(sim_matrix_trunc[14])
val2 = np.partition(QH_edges.flatten(), -3)[-3]
QH_edges[QH_edges < val2] = 0
for k in range(CONST_NUM_PROV):
    sim_matrix_trunc[14, k] = QH_edges[k]
    sim_matrix_trunc[k, 14] = sim_matrix_trunc[14, k]
sim_matrix_trunc[14, 14] = 0


# add edges according to coordinates of adj_matrix
# https://stackoverflow.com/questions/4288973/whats-the-difference-between-s-and-d-in-python-string-formatting
for i in range(CONST_NUM_PROV):
    for j in range(i, CONST_NUM_PROV):
        # G0- visualize edges for neighboring provinces
        if (prc_prov_adj_matrix_[i, j] == 1):
            G0.add_edge("%d" % i, "%d" % j, weight=prc_prov_similarity_matrix[i, j])
        # G1- visualize all edges > 10
        if (prc_prov_similarity_matrix[i, j] >= 10):
            G1.add_edge("%d" % i, "%d" % j, weight=prc_prov_similarity_matrix[i, j])
        # G2- visualize highest value edges > 10
        if (sim_matrix_trunc[i, j] >= 10):
            G2.add_edge("%d" % i, "%d" % j, weight=prc_prov_similarity_matrix[i, j])

# https://stackoverflow.com/questions/3567018/how-can-i-specify-an-exact-output-size-for-my-networkx-graph
# https://networkx.org/documentation/stable/reference/drawing.html
# https://stackoverflow.com/questions/20381460/networkx-how-to-show-node-and-edge-attributes-in-a-graph-drawing


'''
LatLong to Coordinates Conversion...
Y-axis: 15 to 45 N
X-axis: 80 to 130 E
'''
'''
pos = {
    "0": (9.0, 9.3),
    "1": (7.7, 7.1),
    "2": (7.3, 8.3),
    "3": (7.3, 8.1),
    "4": (6.5, 7.6),
    "5": (6.6, 9.7),
    "6": (7.4, 8.0),
    "7": (6.7, 6.3),
    "8": (5.7, 6.9),
    "9": (4.4, 7.7),
    "10": (5.3, 7.8),
    "11": (1.0, 8.7),
    "12": (7.4, 5.6),
    "13": (8.0, 6.0),
    "14": (3.2, 6.7),
    "15": (6.5, 5.4),
    "16": (5.3, 4.9),
    "17": (4.8, 5.2),
    "18": (5.4, 3.9),
    "19": (6.6, 4.4),
    "20": (4.1, 3.4),
    "21": (7.2, 4.1),
    "22": (8.3, 5.4),
    "23": (8.1, 4.7),
    "24": (7.7, 3.6),
    "25": (6.7, 2.8),
    "26": (5.9, 1.4),
    "27": (5.7, 2.9)
}
'''

# tweaked positions, based on real-life latitude and longitude
pos = {
    "0": (9.0, 9.3),
    "1": (7.7, 7.1),
    "2": (8.4, 8.0),
    "3": (7.3, 7.9),
    "4": (6.5, 7.6),
    "5": (6.6, 8.7),
    "6": (8.6, 7.4),
    "7": (6.7, 6.3),
    "8": (5.7, 6.4),
    "9": (4.4, 7.7),
    "10": (5.3, 7.8),
    "11": (1.0, 8.7),
    "12": (7.4, 5.6),
    "13": (8.0, 6.0),
    "14": (3.2, 6.7),
    "15": (6.5, 5.4),
    "16": (5.3, 4.9),
    "17": (4.6, 5.2),
    "18": (5.3, 4.0),
    "19": (6.4, 4.4),
    "20": (4.1, 3.4),
    "21": (7.2, 4.6),
    "22": (8.3, 5.4),
    "23": (8.1, 4.7),
    "24": (7.7, 3.6),
    "25": (6.7, 2.8),
    "26": (5.9, 1.8),
    "27": (5.7, 2.9)
}

f = plt.figure(1, figsize=(12,12))
nx.draw(G0, pos, node_color="red", node_size=1000, with_labels=True, font_color="white", font_size=20)
labels = {e: G0.edges[e]['weight'] for e in G0.edges}
nx.draw_networkx_edge_labels(G0, pos, edge_labels=labels, font_size=14)
f.show()

g = plt.figure(2, figsize=(12,12))
nx.draw(G1, pos, node_color="blue", node_size=1000, with_labels=True, font_color="white", font_size=20)
g.show()

h = plt.figure(2, figsize=(12,12))
nx.draw(G2, pos, node_color="green", node_size=1000, with_labels=True, font_color="white", font_size=20)
h.show()


# https://stackoverflow.com/questions/52251556/remove-all-edges-from-a-graph-in-networkx
G0.remove_edges_from(G0.edges())
G1.remove_edges_from(G1.edges())
G2.remove_edges_from(G2.edges())
