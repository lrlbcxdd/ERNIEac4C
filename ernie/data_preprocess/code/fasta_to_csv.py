from Bio import SeqIO
import pandas as pd

# 设置文件路径
fasta_file = "/home/lrl/ac4C/data_preprocess/fasta/neg_1001.fasta"
csv_file = "/home/lrl/ac4C/data_preprocess/seq_csv/neg_seq.csv"

# 读取FASTA文件并提取序列
sequences = []
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append({"sequence": str(record.seq), "label": 0})

# 将序列数据存入DataFrame
df = pd.DataFrame(sequences)

# 将DataFrame保存为CSV文件
df.to_csv(csv_file, index=False)

print(f"CSV文件已保存为: {csv_file}")
