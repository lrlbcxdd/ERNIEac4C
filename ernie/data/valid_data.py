import pandas as pd
neg_data = pd.read_csv("/mnt/sdb/home/lrl/code/new_ac4c/ernie/data/new_test_data/all_data/neg.csv")
sequences = list(neg_data['Sequence'])

valid_data = pd.read_csv("/mnt/sdb/home/lrl/code/new_ac4c/ernie/data/new_test_data/all_data/pos.csv")
test_seq = list(valid_data['Sequence'])

count = 0
for seq in sequences:
    if seq[207] == 'C':
        count += 1

print(len(sequences))
print(count)