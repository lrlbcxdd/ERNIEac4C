import pandas as pd

# 读取原始CSV文件
file_path = '/home/lrl/ac4C/data_preprocess/seq_csv/neg_seq.csv'  # 请替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 随机选择1850条记录
df_sampled = df.sample(n=1850, random_state=42)

# 将抽样后的数据保存到新的CSV文件
df_sampled.to_csv('/home/lrl/ac4C/data_preprocess/seq_csv/sampled_data.csv', index=False)

print("抽样完成，已保存为 'sampled_data.csv'")
