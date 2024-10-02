import pandas as pd

# 读取CSV文件
# df_trainpos = pd.read_csv('/mnt/sdb/home/lrl/code/Dataset/qjb/train/18.csv')
# df_trainneg = pd.read_csv('/mnt/sdb/home/lrl/code/Dataset/qjb/train/19.csv')
# df_valpos = pd.read_csv('/mnt/sdb/home/lrl/code/Dataset/qjb/val/18.csv')
# df_valneg = pd.read_csv('/mnt/sdb/home/lrl/code/Dataset/qjb/val/19.csv')
# df_testpos = pd.read_csv('/mnt/sdb/home/lrl/code/Dataset/qjb/test/18.csv')
# df_testneg = pd.read_csv('/mnt/sdb/home/lrl/code/Dataset/qjb/test/19.csv')
#
# df1 = pd.read_csv("/mnt/sdb/home/lrl/code/ac4c/data/all_data/minus_2/minus_pos.csv")
# df2 = pd.read_csv("/mnt/sdb/home/lrl/code/ac4c/data/all_data/minus_2/minus_neg.csv")

test_file = pd.read_csv("/mnt/sdb/home/lrl/code/ac4c/data/new_test_data/all_data/balance_data.csv")

middle_count = {}
# 定义函数来统计中间字符的种类和数目
def count_middle_characters(string):
    length = len(string)
    middle_index = length // 2

    # 如果字符串长度为偶数
    if length % 2 == 0:
        middle_characters = string[middle_index - 1: middle_index + 1]
    else:
        middle_characters = string[middle_index]

    # 统计中间字符的种类和数目

    for char in middle_characters:
        if char in middle_count:
            middle_count[char] += 1
        else:
            middle_count[char] = 1

# for i in list(df_trainpos['data']):
#     count_middle_characters(i)
# for i in list(df_trainneg['data']):
#     count_middle_characters(i)
# for i in list(df_valpos['data']):
#     count_middle_characters(i)
# for i in list(df_valneg['data']):
#     count_middle_characters(i)
# for i in list(df_testpos['data']):
#     count_middle_characters(i)
# for i in list(df_testneg['data']):
#     count_middle_characters(i)
# for i in list(df1['Sequence']):
#     count_middle_characters(i)
# for i in list(df2['Sequence']):
#     count_middle_characters(i)
for i in list(test_file['Sequence']):
    count_middle_characters(i)
# 显示结果
print("中间碱基及数目：",middle_count)
print()
#
# count = 0
# for i in list(df_trainpos['data']):
#     if i.find('N') <100:
#         # print(i)
#         count += 1
# for i in list(df_trainneg['data']):
#     if i.find('N') <100:
#         # print(i)
#         count += 1
# for i in list(df_valpos['data']):
#     if i.find('N') <100:
#         # print(i)
#         count += 1
# for i in list(df_valneg['data']):
#     if i.find('N') <100:
#         # print(i)
#         count += 1
# for i in list(df_testpos['data']):
#     if i.find('N') <100:
#         # print(i)
#         count += 1
# for i in list(df_testneg['data']):
#     if i.find('N') <100:
#         # print(i)
#         count += 1
# print("左边有N的序列条数：",count)    #1
#
#
# max_len = 0
# min_len = 100000
# for i in list(df_trainpos['data']):
#     s = i.rstrip('N')
#     if len(s) > max_len : max_len = len(s)
#     if len(s) < min_len : min_len = len(s)
# for i in list(df_trainneg['data']):
#     s = i.rstrip('N')
#     if len(s) > max_len : max_len = len(s)
#     if len(s) < min_len : min_len = len(s)
# for i in list(df_valpos['data']):
#     s = i.rstrip('N')
#     if len(s) > max_len : max_len = len(s)
#     if len(s) < min_len : min_len = len(s)
# for i in list(df_valneg['data']):
#     s = i.rstrip('N')
#     if len(s) > max_len : max_len = len(s)
#     if len(s) < min_len : min_len = len(s)
# for i in list(df_testneg['data']):
#     s = i.rstrip('N')
#     if len(s) > max_len : max_len = len(s)
#     if len(s) < min_len : min_len = len(s)
# for i in list(df_testpos['data']):
#     s = i.rstrip('N')
#     if len(s) > max_len : max_len = len(s)
#     if len(s) < min_len : min_len = len(s)
#
# print()
# print("去掉右侧N之后的序列长度")
# print("max_len:",max_len, "min_len:",min_len)
#
#
# def calculate_substring_length(string):
#     # 计算中间字符的位置
#     middle_index = len(string) // 2
#
#     # 从中间字符的位置开始，统计到右边没有N的子字符串的长度
#     substring_length = 0
#     for i in range(middle_index, len(string)):
#         if string[i] == 'N':
#             break
#         substring_length += 1
#
#     return substring_length
#
# right_max = 0
# right_min = 10000
# for i in list(df_trainpos['data']):
#     right_length = calculate_substring_length(i)
#     if right_length > right_max : right_max = (right_length)
#     if right_length < right_min : right_min = (right_length)
#
# for i in list(df_trainneg['data']):
#     right_length = calculate_substring_length(i)
#     if right_length > right_max : right_max = (right_length)
#     if right_length < right_min : right_min = (right_length)
#
# for i in list(df_valpos['data']):
#     right_length = calculate_substring_length(i)
#     if right_length > right_max : right_max = (right_length)
#     if right_length < right_min : right_min = (right_length)
#
# for i in list(df_valneg['data']):
#     right_length = calculate_substring_length(i)
#     if right_length > right_max : right_max = (right_length)
#     if right_length < right_min : right_min = (right_length)
#
# for i in list(df_testpos['data']):
#     right_length = calculate_substring_length(i)
#     if right_length > right_max : right_max = (right_length)
#     if right_length < right_min : right_min = (right_length)
#
# for i in list(df_testneg['data']):
#     right_length = calculate_substring_length(i)
#     if right_length > right_max : right_max = (right_length)
#     if right_length < right_min : right_min = (right_length)
#
# print()
# print("右侧长度")
# print("right_min:",right_min,"right_max:",right_max)