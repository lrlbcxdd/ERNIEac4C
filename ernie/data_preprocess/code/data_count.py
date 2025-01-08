def parse_fasta(file_path):
    """
    解析FASTA文件并返回序列列表和对应的序列名称
    """
    sequences = []
    sequence = ""
    seq_name = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append((seq_name, sequence))  # 保存前一条序列
                seq_name = line[1:]  # 获取序列名（去掉 >）
                sequence = ""  # 重置序列
            else:
                sequence += line  # 将序列拼接起来
        if sequence:
            sequences.append((seq_name, sequence))  # 最后一条序列

    return sequences


def analyze_sequences(file_path):
    sequences = parse_fasta(file_path)

    # 统计序列的数目
    num_sequences = len(sequences)
    print(f"总共有 {num_sequences} 条序列。")

    for seq_name, sequence in sequences:
        # 统计每条序列的长度
        seq_length = len(sequence)
        print(f"序列 {seq_name} 的长度是 {seq_length}。")

        # 统计中间位点的字符
        middle_index = seq_length // 2
        if seq_length % 2 == 0:
            middle_char = sequence[middle_index - 1:middle_index + 1]  # 偶数长度，取中间两个字符
            print(f"序列 {seq_name} 中间位点的字符是：{middle_char}")
        else:
            middle_char = sequence[middle_index]  # 奇数长度，取中间一个字符
            print(f"序列 {seq_name} 中间位点的字符是：{middle_char}")


# 输入文件路径
file_path = '/home/lrl/ac4C/data_preprocess/fasta/pos_1001.fasta'

# 执行统计和分析
analyze_sequences(file_path)
