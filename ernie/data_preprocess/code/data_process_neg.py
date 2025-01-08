import numpy as np
from pyfaidx import Fasta
import pandas as pd
fasta_file = Fasta('/mnt/8t/jjr/chip_plot/GRCH38/GRCh38.p13.genome.fa')


# 负例
neg_sites = [24000000, 25000000, 16000000, 11000000, 11000000, 11000000, 28000000, 20000000, 20000000,
             36000000, 15000000, 11000000, 26000000, 16000000, 16000000, 16000000, 24000000, 18000000,
             21000000, 14000000, 21000000, 14000000, 14000000]
counts = [900, 500, 400, 200, 600, 350, 400, 300, 450, 200, 600, 650, 100, 250, 250, 350, 700, 100, 700,
          200, 100, 250, 350]
names = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
        'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

with open('neg_1001.fasta', 'a') as file:
    for i in range(len(names)):
        neg_site = neg_sites[i]
        count = counts[i]
        name = names[i]
        current_count = 1
        end_site = neg_site + 1000000
        while neg_site <= end_site:
            if current_count > count:
                current_count = 1
                break
            nt = fasta_file[name][neg_site]
            if nt == 'C':
                item = fasta_file[name][neg_site-500:neg_site + 501]
                neg_site += 1001

                file.write(f'>neg_{name}_{neg_site}\n')
                file.write(f'{item.seq}\n')
                current_count += 1
            neg_site += 1