import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/lrl/ac4C/data_preprocess/seq_csv/balanced_data.csv')

# 初始化两个字典
middle_nucleotide_count = {}  # 统计中间位点的类别
sequence_length_count = {}  # 统计序列长度的频次

# 遍历所有行，获取中间位点和序列长度
for sequence in df['sequence']:
    # 获取中间位点的索引
    middle_index = len(sequence) // 2  # 如果长度是偶数，则取靠左的中间位
    middle_nucleotide = sequence[middle_index]

    # 更新中间位点的计数
    if middle_nucleotide in middle_nucleotide_count:
        middle_nucleotide_count[middle_nucleotide] += 1
    else:
        middle_nucleotide_count[middle_nucleotide] = 1

    # 获取序列长度
    sequence_length = len(sequence)

    # 更新序列长度的计数
    if sequence_length in sequence_length_count:
        sequence_length_count[sequence_length] += 1
    else:
        sequence_length_count[sequence_length] = 1

# 输出统计结果
print("中间位点的统计：")
print(middle_nucleotide_count)

print("\n序列长度的统计：")
print(sequence_length_count)
