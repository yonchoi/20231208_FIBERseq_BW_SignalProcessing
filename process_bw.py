import os,glob

import numpy as np
import pandas as pd

from typing import Optional

from tqdm import tqdm
from src.seq2annot.util.naming import add_suffix

from functools import reduce

import pyBigWig
import pyfaidx

def get_chr2len(file):
    filetype = get_track_filetype(file)
    if filetype == 'fasta':
        fa = pyfaidx.Fasta(file)
        chr2len = {chr:len(fa[chr]) for chr in fa.keys()}
    elif filetype == 'bigwig':
        bw = pyBigWig.open(file)
        chr2len = bw.chroms()
    else:
        raise Exception('Invalid file type')
    return chr2len


def get_track_filetype(FILE_DIR):
    print(FILE_DIR)
    ext = os.path.splitext(FILE_DIR)[1].strip(".").lower()
    if ext in ['bw','bigwig']:
        filetype = 'bigwig'
    elif ext in ['fasta','fa','fas']:
        filetype = 'fasta'
    elif ext in ['bigbed']:
        filetype = 'bigbed'
    elif ext in ['tsv','csv']:
        filetype = 'matrix'
    else:
        raise Exception(f"Input FILE_DIR with valid extension, {ext}")
    return filetype


def create_template_bw(
    bwfile: str = '/tmp/tempfile.bw',
    REF_DIR: str = None,
    chr2len: dict = None,
    keep_chr_only: bool = False,
):
    # Set up length per chromosome
    if chr2len is None:
        chr2len = get_chr2len(REF_DIR)
    if keep_chr_only:
        chr2len = {k:v for k,v in chr2len.items() if k.startswith('chr')}
    # Create bigwig file
    bw = pyBigWig.open(bwfile, "w")
    bw.addHeader([(k,v) for k,v in chr2len.items()])
    return bw


class Bed():
    def __init__(self,chr=None,length=10000,default_value=0.0):
        self.raw = np.ones(length)*default_value
        self.chr = chr
    def set_raw(self,raw):
        self.raw = raw
    def update(self,start,end,value=1.0):
        self.raw[start:end]=value
    def update_with_df(self,df,value=1.0):
        for start,end in zip(df.start,df.end):
            self.update(start,end,value)
    def export_bed(self):
        """
        output in bed format that can be used for bigwig
        """
        starts = []
        ends   = []
        values = []
        for i,v in tqdm(enumerate(self.raw), total=len(self.raw)):
            if i == 0:
                start = i
                value = v
            #
            if value != v:
                starts.append(start)
                ends.append(i)
                values.append(value)
                start = i
                value = v
        starts.append(start)
        ends.append(i)
        values.append(v)
        return [self.chr]*len(starts),starts,ends,values

def get_chr_intersect(chr2len_1, chr2len_2):
    """ Take in dictionaries mapping chr name to its length, return overlap """
    chr_intersect = np.intersect1d(list(chr2len_1.keys()),list(chr2len_2.keys()))
    chr2len = {}
    for chr in chr_intersect:
        maxlen = np.min([chr2len_1[chr],chr2len_2[chr]])
        chr2len[chr] = maxlen
    return chr2len

def perform_bw_op(
    bwfile_out: str,
    bwfiles_in: list,
    operation: str == 'add',
    keep_chr_only: bool = True,
    weights: Optional[list] = None
    ):
#
    # Get intersection of all chr across bwfiles
    chr2len_list = [get_chr2len(bwfile) for bwfile in bwfiles_in]
#
    # Keep only chromosomes starting with chr
    if keep_chr_only:
        chr2len_list = [{k:v for k,v in chr2len.items() if k.startswith('chr')} for chr2len in chr2len_list]
#
    chr2len = reduce(get_chr_intersect, chr2len_list)
#
    # Load input bw files
    bw_list = [ pyBigWig.open(bwfile) for bwfile in bwfiles_in]
#
    # Create bw file that result will be written to
    bw = create_template_bw(bwfile_out, chr2len=chr2len)
#
    if weights is None:
        weights = [1] * len(bw_list)
#
    if operation == 'mean':
        weights = np.array(weights) / np.sum(weights)
#
    assert len(bw_list) == len(weights), "# of bwfiles and weights do not match"
#
    # Perform operation per chr
    for chr in tqdm(chr2len.keys(), total=len(chr2len)):
#
        chr_length = chr2len[chr]
#
        # for i,bw_in, w in tqdm(enumerate(zip(bw_list,weights))):
        #     # Load the bw file as numpy array
        #     raw = np.nan_to_num(np.array(bw_in.values(chr,0,chr_length)))
        #     raw *= w # weight by w
        #     # Perform operation
        #     if i == 0:
        #         track = raw
        #     else:
        #         if operation == 'add':
        #             track += raw
        #         elif operation == 'multiply':
        #             track *= raw
        #         else:
        #             raise ValueError(f'Input valid operation, operation={operation}')
#
        # Gather raws as numpy array
        raws = []
        for i, (bw_in, w) in (enumerate(zip(bw_list,weights))):
            # Load the bw file as numpy array
            raw = np.nan_to_num(np.array(bw_in.values(chr,0,chr_length)))
            raw *= w # weight by w
            raws.append(raw)
#
        # Perform operation
        if operation == 'add':
            track = np.sum(raws, axis=0)
        elif operation == 'multiply':
            track = np.prod(raws, axis=0)
        elif operation == 'ratio':
            track = raws[0] / np.maximum(np.sum(raws, axis=0),1.0)
        else:
            raise ValueError(f'Input valid operation, operation={operation}')
#
        # Export values for bigwig
        bed = Bed(chr,length=chr_length)
        bed.set_raw(track)
        chrs,starts,ends,values = bed.export_bed()
#
        # Add to track for chr to bw
        bw.addEntries(chrs,starts,ends=ends,values=values)
#
    # Close all bigwig files
    bw.close()
    _ = [bw_in.close() for bw_in in bw_list]


def write_bw_for_mA_tracks(OUT_DIR,sample='D2stim',haplo='H1',signal_type='mA_mean'):
    # Set args for signal_type
    if signal_type == 'mA_mean':
        ms = ['m6A','m0A','mxA']
        op = 'ratio'
    elif signal_type == 'mA_ratio':
        ms = ['m6A','m0A']
        op = 'ratio'
    elif signal_type == 'mA_sum':
        ms = ['m6A','m0A']
        op = 'add'
    else:
        raise ValueError(f"Input valid signal_type: {signal_type}")
    # Gather bwfiles
    bwdir=f'/net/seq/data2/projects/Palladium_Dataset/sasha/{sample}_bigwigs'
    bwfiles = []
    for m in ms:
        bwfile = os.path.join(bwdir,f"{sample}{haplo}_{m}.bw")
        assert os.path.isfile(bwfile), f'path {bwfile} does not exist'
        bwfiles.append(bwfile)
    #
    # Write bw file after calculating the mA signal
    bwfile_out = add_suffix(f"{sample}{haplo}_{signal_type}",ext='bw',OUT_DIR=OUT_DIR)
    #
    perform_bw_op(
        bwfile_out = bwfile_out,
        bwfiles_in = bwfiles,
        operation = op,
    )


def parse_chr_str(s):
    chr,rest  = s.split(':')
    start,end = rest.split('-')
    return chr,int(start),int(end)


def test_bw_track(bwfile_out):
    bwfile_out = 'experiment/2023_fiberseq/fiberseq_signal/D2restH1_mA_mean-YC20231208.bw'
    chr_str = 'chr17:42383710-42393710'
    chr,start,end = parse_chr_str(chr_str)
    with pyBigWig.open(bwfile_out) as bw:
        a = bw.values(chr,start,end)
        a = np.array(a)
    return a


if __name__ == '__main__':

    sample='D2stim'
    haplo='H1'

    #### Create tracks
    OUT_DIR = 'experiment/2023_fiberseq/fiberseq_signal'
    os.makedirs(OUT_DIR,exist_ok=True)

    write_bw_for_mA_tracks(OUT_DIR,
                        sample='D2rest',
                        haplo='H1',
                        signal_type='mA_mean')
