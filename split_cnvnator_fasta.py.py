# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pysam

p=argparse.ArgumentParser()
p.add_argument('--fasta',required=True)
p.add_argument('--out',required=True)
args=p.parse_args()
out=Path(args.out)
out.mkdir(parents=True,exist_ok=True)
with pysam.FastaFile(args.fasta) as fasta:
    for number in range(1,23):
        chrom='chr'+str(number)
        if chrom not in fasta.references:
            raise ValueError('Missing reference: '+chrom)
        length=fasta.get_reference_length(chrom)
        with (out/(chrom+'.fa')).open('w',encoding='ascii') as handle:
            handle.write('>'+chrom+'\n')
            for start in range(0,length,600000):
                seq=fasta.fetch(chrom,start,min(start+600000,length)).upper()
                for pos in range(0,len(seq),60):
                    handle.write(seq[pos:pos+60]+'\n')
        print('Saved',chrom,length,flush=True)
