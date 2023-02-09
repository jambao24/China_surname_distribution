# China_10_most_common_surnames_by_province
# 

import pandas as pd
import numpy as np
import pandas.plotting._matplotlib as plt

raw = pd.read_csv('China_top10surname.csv', skiprows=1, names=['Province','1','2','3','4','5','6','7','8','9','10'])
data = np.array(raw)

#print(raw)
#print(data)
print()

'''
Clustering of provinces by surname distribution- 
For each "province", calculate similarity of surname frequency distribution with all other "provinces". 

Similarity metric- 
1) 1 point for each surname (number) found in both the "test" province and the "reference" province (max possible: 10)
2) 1 point for each surname (number) found in the same position in both provinces for indices 4 to 10 (max possible: 6)
3) 2 points for each surname (number) found in the same position in both provinces for indices 2 to 3 (max posssible: 4)
4) 5 points for matching most common surnames (max possible: 5)

Create confusion matrix for all 28 provinces
'''

'''
Map visualization (models the 28 provinces as a minimal spanning tree)

https://www.youtube.com/watch?v=LFT0rj-xvYA for Prim's algorithm

Dec 2022 update- https://www.ecosia.org/search?addon=chrome&addonversion=5.1.2&q=how%20to%20store%20chinese%20characters%20as%20information%20in%20python

Idea- eventually edit each element in the map to be a data structure that contains a province (abbreviated, full English, full Chinese), and a surname 
(ideally in converted format and with the original character being displayed with pinyin also displayed)...

'''

provinces_map = np.chararray((8,6))
provinces_map[:] = '.'
provinces_map[0,2] = 'O' # Xinjiang (XJ)
provinces_map[0,3] = 'O' # InnerMongolia (MON)
provinces_map[1,3] = 'O' # Beijing (BJ)
provinces_map[1,4] = 'O' # Manchuria (MAN)
provinces_map[2,3] = 'O' # Hebei (HEB)
provinces_map[2,4] = 'O' # Tianjin (TJ)
provinces_map[2,1] = 'O' # Ningxia (NX)
provinces_map[2,2] = 'O' # Shanxi (SX)
provinces_map[3,0] = 'O' # Qinghai (QH)
provinces_map[3,1] = 'O' # Gansu (GS)
provinces_map[3,2] = 'O' # Shaanxi (SW)
provinces_map[3,3] = 'O' # Henan (HEN)
provinces_map[3,4] = 'O' # Shandong (SD)
provinces_map[4,1] = 'O' # Sichuan (SC)
provinces_map[4,2] = 'O' # Chongqing (CQ)
provinces_map[4,3] = 'O' # Hubei (HUB)
provinces_map[4,4] = 'O' # Jiangsu (JS)
provinces_map[4,5] = 'O' # Shanghai (SH)
provinces_map[5,1] = 'O' # Guizhou (GUI)
provinces_map[5,2] = 'O' # Hunan (HUN) 
provinces_map[5,3] = 'O' # Jiangxi (JX)
provinces_map[5,4] = 'O' # Anhui (AH))
provinces_map[5,5] = 'O' # Zhejiang (ZJ)
provinces_map[6,1] = 'O' # Yunnan (YUN)
provinces_map[6,3] = 'O' # Guangxi (GX)
provinces_map[6,4] = 'O' # Guangdong (GD)
provinces_map[6,5] = 'O' # Fujian (FJ)
provinces_map[7,4] = 'O' # Hainan (HAN)

print(provinces_map)
print()


# source: https://i.imgur.com/m8M4Dwj.jpg
prc_prov_adj_matrix = 
                    [[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Manchuria
                    [0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Shandong
                    [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Beijing
                    [1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Hebei
                    [0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Shanxi
                    [0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # InnerMongolia
                    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Tianjin
                    [0,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # Henan
                    [0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0], # Shaanxi
                    [0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0], # Gansu
                    [0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Ningxia
                    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # Xinjiang
                    [0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0], # Anhui
                    [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0], # Jiangsu
                    [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], # Qinghai
                    [0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0], # Hubei
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0], # Chongqing
                    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0], # Sichuan
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,1], # Guizhou
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1], # Hunan
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1], # Yunnan
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0], # Jiangxi
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0], # Shanghai
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0], # Zhejiang
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0], # Fujian
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,1], # Guangdong
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # Hainan
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0]] # Guangxi
                    
                    
# https://www.youtube.com/watch?v=HDUzBEG1GlA
# https://github.com/joeyajames/Python/blob/master/graph_adjacency-matrix.py

# implementation of an undirected graph using adjacency matrix
class Vertex:
  def __init__(self, n):
    self.name = n
    self.

class Graph:
  vertices = {}
  edges = {}
  edge_indices = {}


# https://www.linkedin.com/pulse/beginners-guide-choropleth-map-python-visualize-chinas-ying-li/
# for visualization of China on an actual map?




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
      tally += 2
  return tally


# Python function for similarity metric 4
# comparing the elements of 2 arrays from indices 1 to 2, assuming both have a length of 10
def identical_1(a, b):
  if (len(a) != len(b) or len(a) != 10):
    return -1
  if a[0] == b[0]:
      return 5
  return 0