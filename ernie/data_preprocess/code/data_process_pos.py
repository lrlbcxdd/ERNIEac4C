import numpy as np
from pyfaidx import Fasta
import pandas as pd
fasta_file = Fasta('/mnt/8t/jjr/chip_plot/GRCH38/GRCh38.p13.genome.fa')

data_indexes = pd.read_csv('/home/lrl/ac4C/data_preprocess/ac4c_HEK293T.csv')

ModChrs = data_indexes['ModChr']
sites = data_indexes['ModStart']
Strands = data_indexes['Strand']

with open('pos_1001.fasta', 'a') as file:
    for i in range(len(ModChrs)):
        chromosome = ModChrs[i]
        if type(chromosome) is not str:
            break
        site = int(sites[i])
        start = site-500
        end = site+501
        strand = Strands[i]
        item = fasta_file[chromosome][start:end]
        if strand == '+':
            seq = item.seq
        elif strand == '-':
            seq = item.complement.reverse
        name = item.name
        file.write(f'>{name}:{start}:{end} {strand}\n')
        file.write(f'{seq}\n')