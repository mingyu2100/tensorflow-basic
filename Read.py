import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import pandas as pd
from scipy.misc import imread, imsave, imresize
from scipy.ndimage.interpolation import rotate
import codecs


# 각 지점(위,경도)의 AP 갯수 & 모든 지점 중 MAX,MIN,AVERAGE 구하기

# data_file = open('wifi_raw_201803191724.csv', 'r')
# data_file = open('wifi_T1_-1.csv', 'r')
# data_file = open('wifi_T1_1.csv', 'r')
# data_file = open('wifi_T1_3.csv', 'r')
# data_file = open('wifi_T2_-1.csv', 'r')
# data_file = open('wifi_T2_1.csv', 'r')
# data_list = data_file.readlines()
# data_file.close()
#
# total = []
# for record in data_list:
#     total.append((record.split(',')[0], record.split(',')[1][:-1]))
#
# _total = set(total)
# # print(len(_total))
#
# _list = []
# AP_num = []
# for y in _total:
#     temp = []
#     check = 0
#     _list.append(y)
#     for x in total:
#         if y == x:
#             check += 1
#     AP_num.append(check)
#     _list.append(check)
#
# check = 0
# for i in _list:
#     print(i, end=' ')
#     check += 1
#     if check == 2:
#         check = 0
#         print(end='\n')
#
# # print(_list)
# # print(AP_num)
#
# _max = max(AP_num)
# _min = min(AP_num)
# _mean = np.mean(AP_num)
# print('max: ', _max)
# print('min: ', _min)
# print('mean: ', _mean)

# 각 지점(위, 경도)의 AP 세기(RSSI) 구하기

data_file = open('T1\GalaxyA\Evening2.csv', 'r')
data_list = data_file.readlines()[1:]
data_file.close()

data = []
for i in data_list:
    data.append(i.split(','))
print('Longitude: {}, Latitude: {}, RSSI: {}'.format(data[0][10], data[0][11], data[0][15]))

array_data = np.array(data)
Longitude = set(array_data[:, 10])
Latitude = set(array_data[:, 11])

AP = []
for i in Longitude:
    t = []
    t.append(i)
    for j in array_data:
        if j[10] == i:
            t.append(j[15][1:3])
    AP.append(t)

for i in AP:
    print(i)
    plt.plot(range(len(i[1:])), i[1:])
    plt.show()








# total = []
# longitude = []
# latitude = []
# for record in data_list:
#     longitude.append(record.split(',')[0])
#     latitude.append(record.split(',')[1][:-1])
#     total.append((record.split(',')[0], record.split(',')[1][:-1]))
# # print(longitude)
# # print(latitude)
# # print(total)
#
# remove_dup_longitude = set(longitude)
# remove_dup_latitude = set(latitude)
# # print(remove_dup_longitude)
# # print(remove_dup_latitude)
#
# wifi_array = np.array(np.zeros((len(remove_dup_longitude), len(remove_dup_latitude))))
#
# temp = []
# for idx, x in enumerate(remove_dup_longitude):
#     line = []
#     line.append(x)
#     for y in total:
#         if x == y[0]:
#             line.append(y[1])
#     temp.append(line)
# print(temp)
#


