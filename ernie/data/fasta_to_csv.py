import csv
import pandas as pd
import random
def read_fasta_file(file_path):
    sequences = []
    headers = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i in range(0,len(lines),2):
            line = lines[i].strip()

            if line.startswith('>'):
                header = line[1:]
                headers.append(header)
                sequences.append(lines[i+1].strip())

    return headers, sequences


def write_to_csv(sequences):
    data = []

    for sequence in sequences:
        entry = [sequence, 0]
        data.append(entry)

    # with open('/mnt/sdb/home/lrl/code/new_ac4c/ernie/data/new_test_data/all_data/neg.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)


# pos_file = '/mnt/sdb/home/lrl/code/ac4c/data/new_test_data/test_fasta/pos.fasta'
neg_file = '/mnt/sdb/home/lrl/code/ac4c/data/new_test_data/test_fasta/neg.fasta'
headers, sequences = read_fasta_file(neg_file)
print(len(sequences))
# sequences = random.sample(sequences,1850)
write_to_csv(sequences)