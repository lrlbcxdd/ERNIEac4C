import numpy as np
from pyfaidx import Fasta
import pandas as pd
fasta_file = Fasta('/mnt/8t/jjr/chip_plot/GRCH38/GRCh38.p13.genome.fa')

# data_indexes = pd.read_csv('m7g-seq.csv')

# ModChrs = data_indexes['seqnames']
# sites = data_indexes['start']
# Strands = data_indexes['strand']

idx = 1
# with open('pos.fasta', 'r') as file:
#     for line in file:
#         if line.startswith('>'):
#             continue
#         if line[20] != 'G':
#             print(line[20])
#         idx+=1
#         print(line[19])


neg_sites = [24000000, 25000000, 16000000, 11000000, 11000000, 11000000, 28000000, 20000000, 20000000,
             36000000, 15000000, 11000000, 26000000, 16000000, 16000000, 16000000, 24000000, 18000000,
             21000000, 14000000, 21000000, 14000000, 14000000]
# value_to_subtract = 237000000
# modified_numbers = [num + value_to_subtract for num in neg_sites]
counts = [900, 500, 400, 200, 600, 350, 400, 300, 450, 200, 600, 650, 100, 250, 250, 350, 700, 100, 700,
          200, 100, 250, 350]
names = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
        'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
with open('neg.fasta', 'a') as file:
    for i in range(len(names)):
        neg_site = neg_sites[i] # 1000000
        count = counts[i]   # 2000000
        name = names[i]
        current_count = 1
        end_site = neg_site + 1000000
        while neg_site <= end_site:
            if current_count > count:
                current_count = 1
                break
            nt = fasta_file[name][neg_site]
            if nt == 'C':
                item = fasta_file[name][neg_site-207:neg_site + 208]
                neg_site += 415

                file.write(f'>neg_{name}_{neg_site}\n')
                file.write(f'{item.seq}\n')
                current_count += 1
            neg_site += 1

# with open('pos_501.fasta', 'w') as file:
#     for i in range(len(ModChrs)):
#         chromosome = ModChrs[i]
#         if type(chromosome) is not str:
#             break
#         site = int(sites[i])
#         start = site-251
#         end = site+250
#         strand = Strands[i]
#         item = fasta_file[chromosome][start:end]
#         if strand == '+':
#             seq = item.seq
#         elif strand == '-':
#             seq = item.complement.reverse
#         name = item.name
#         file.write(f'>{name}:{start}:{end} {strand}\n')
#         file.write(f'{seq}\n')
