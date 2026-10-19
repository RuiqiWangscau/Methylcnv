# -*- coding: utf-8 -*-
"""Merge legacy Pindel VCF text before parsing records with HTSlib.

Adds reference contig declarations and repairs AD/PL cardinality declarations.
Preserves record text, genotypes and coordinates. Output is sorted downstream.
"""
import argparse
import re
from pathlib import Path


def repair_format(line):
    for tag,number in [('AD','R'),('PL','G')]:
        if line.startswith('##FORMAT=<ID='+tag+','):
            return re.sub(r'(?<=,)Number=[^,>]+','Number='+number,line,count=1)
    return line


def merge(directory,fai,output):
    lengths={}
    with open(fai,encoding='utf-8') as handle:
        for line in handle:
            fields=line.split('\t')
            lengths[fields[0]]=int(fields[1])
    paths=[Path(directory)/('chr'+str(i))/'calls.vcf' for i in range(1,23)]
    metadata=[]
    seen=set()
    definitions={}
    columns=None
    version=None
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        found_columns=False
        with path.open(encoding='utf-8') as handle:
            for raw in handle:
                line=raw.rstrip('\r\n')
                if line.startswith('##fileformat='):
                    if version is not None and version!=line:
                        raise ValueError('Different VCF versions')
                    version=line
                elif line.startswith('##contig='):
                    continue
                elif line.startswith('##'):
                    line=repair_format(line)
                    match=re.match(r'##(INFO|FORMAT|FILTER|ALT)=<ID=([^,>]+)',line)
                    if match:
                        key=match.groups()
                        if key in definitions and definitions[key]!=line:
                            raise ValueError('Conflicting VCF definition: '+str(key))
                        definitions[key]=line
                    if line not in seen:
                        seen.add(line)
                        metadata.append(line)
                elif line.startswith('#CHROM\t'):
                    if columns is not None and columns!=line:
                        raise ValueError('Different sample columns: '+str(path))
                    columns=line
                    found_columns=True
                    break
                elif line and not line.startswith('#'):
                    raise ValueError('Missing VCF column header: '+str(path))
        if not found_columns:
            raise ValueError('No #CHROM header: '+str(path))
    if version is None or columns is None:
        raise ValueError('Missing VCF headers')
    count=0
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial')
    with temporary.open('w',encoding='utf-8') as writer:
        writer.write(version+'\n')
        for line in metadata:
            writer.write(line+'\n')
        for chrom,length in lengths.items():
            writer.write(f'##contig=<ID={chrom},length={length}>\n')
        writer.write(columns+'\n')
        for number,path in enumerate(paths,1):
            with path.open(encoding='utf-8') as reader:
                for line in reader:
                    if not line.strip() or line.startswith('#'):
                        continue
                    fields=line.rstrip('\r\n').split('\t')
                    if len(fields)!=len(columns.split('\t')):
                        raise ValueError('VCF record column count mismatch: '+str(path))
                    if fields[0]!='chr'+str(number):
                        raise ValueError('Unexpected record chromosome: '+str(path))
                    if fields[0] not in lengths or not 1<=int(fields[1])<=lengths[fields[0]]:
                        raise ValueError('VCF position outside reference')
                    writer.write(line.rstrip('\r\n')+'\n')
                    count+=1
    temporary.replace(output)
    print(f'Merged {count} records with one complete header: {output}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',required=True)
    p.add_argument('--fasta',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    merge(args.directory,args.fasta+'.fai',args.output)


if __name__=='__main__':
    main()
