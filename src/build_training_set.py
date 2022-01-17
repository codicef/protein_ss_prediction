#!/usr/bin/env python3

import pickle
from parsing_utils import Parser
import argparse
import os

# Parsing arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('list_id', type=str, nargs='+',
                    help='List of proteins id')
parser.add_argument('--ss_directory', type=str, nargs='?',
                    help='Secondary structure directory', default="../data/dssp/")
parser.add_argument('--profiles_directory', type=str, nargs='?',
                    help='Profiles directory', default="../data/profiles/")
parser.add_argument('--fasta_directory', type=str, nargs='?',
                    help='Profiles directory', default="../data/fasta/")
parser.add_argument('--parse_dssp', action='store_const',
                    const=True, default=False,
                    help='Perform parsing of DSSP Files')
parser.add_argument('--keep_empty_profiles', action='store_const',
                    const=True, default=False,
                    help='If empty profiles is found, return one-hot profile')
parser.add_argument('--compute_profiles', action='store_const',
                    const=True, default=False,
                    help='If empty profiles is found, try to compute it')
parser.add_argument('--limit', type=int, nargs='?',
                    help='Limit n entries', default=0)
parser.add_argument('--out', type=str, nargs='?',
                    help='Output file', default="./dataset.pickle")

args = parser.parse_args()
print(args)
list_id = args.list_id[0]
ss_dir = args.ss_directory
profiles_dir = args.profiles_directory
fasta_dir = args.fasta_directory
parse_dssp = args.parse_dssp
keep_empty_profiles = args.keep_empty_profiles
compute_profiles = args.compute_profiles
limit = args.limit
out_file_name = args.out
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']



list_ids = Parser.parse_list_ids(list_id)

dataset = {}
for i, protein_id in enumerate(list_ids):
    if i > limit:
        break
    # Read sequence
    seq = Parser.parse_fasta(protein_id, fasta_directory=fasta_dir)[1]

    # Read profile
    if keep_empty_profiles:
        profile, flag = Parser.parse_profile(protein_id, seq=seq, profiles_directory=profiles_dir)
        if seq is None:
            assert False
    else:
        profile, flag = Parser.parse_profile(protein_id, seq=None, profiles_directory=profiles_dir)
    if profile is None and compute_profiles:
        print("Trying to compute profile for " + protein_id)
        os.system("psiblast -query " + fasta_dir + "" + protein_id +
                  ".fasta -db ../data/databases/uniprot_sprot.fasta -evalue 0.01 -num_iterations 3 \
                  -out_ascii_pssm " + profiles_dir + "" + protein_id + ".pssm -num_descriptions 1000\
                  -num_alignments 1000 -out ../data/alignments/" + protein_id + ".blast")

        if keep_empty_profiles:
            profile, flag = Parser.parse_profile(protein_id, seq=seq, profiles_directory=profiles_dir)
        else:
            profile, flag = Parser.parse_profile(protein_id, seq=None, profiles_directory=profiles_dir)
    if profile is None:
            continue
    # Read secondary structure
    if parse_dssp:
        chains = [a.split("_") for a in list_ids]
        chains = dict(chains)
        ss, seq = Parser.parse_secondary_structure(protein_id, ss_folder=ss_dir, ss_type='dssp',
                                                   chains=chains)
    else:
        ss = Parser.parse_secondary_structure(protein_id, ss_folder=ss_dir, ss_type='fasta')

    assert len(ss) == len(seq) and len(seq) == profile.shape[0]

    dataset[protein_id] = {'seq': seq, 'ss': ss, 'profile': profile, 'empty_profile': flag}

print("Dataset number of elements : " + str(len(dataset.keys())))
with open(out_file_name, "wb") as f:
    pickle.dump(dataset, f)
    print("Training set saved")
