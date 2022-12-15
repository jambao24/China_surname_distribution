# China_10_most_common_surnames_by_province
# 

import pandas as pd
import numpy as np
import pandas.plotting._matplotlib as plt

raw = pd.read_csv('China_top10surname.csv', skiprows=1, names=['Province','1','2','3','4','5','6','7','8','9','10'])
data = np.array(raw)

#rint(raw)
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