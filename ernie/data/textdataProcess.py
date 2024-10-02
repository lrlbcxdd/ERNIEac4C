import pandas as pd

# 读取 CSV 文件
df1 = pd.read_csv(r'D:\课题\final_fw\qjbtest\train\19.csv')
df2= pd.read_csv(r'D:\课题\final_fw\qjbtest\test\19.csv')
df3 = pd.read_csv(r'D:\课题\final_fw\qjbtest\val\19.csv')

# 将指定列的数值全部设置为1
column_name = 'label'  # 你的标签列的名称
df1[column_name] = 1
df2[column_name] = 1
df3[column_name] = 1

# 保存修改后的数据到新的 CSV 文件中
df1.to_csv(r'D:\课题\final_fw\qjbtest\train\191.csv', index=False)
df2.to_csv(r'D:\课题\final_fw\qjbtest\test\191.csv', index=False)
df3.to_csv(r'D:\课题\final_fw\qjbtest\val\191.csv', index=False)
