#!/usr/bin/env python3
from os.path import join, isfile
import pandas as pd
import numpy as np
from Bio.PDB.DSSP import make_dssp_dict

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

class Parser(object):

    @staticmethod
    def parse_profile(protein_id, seq=None, profiles_directory="./profiles/"):
        flag = False
        if isfile(join(profiles_directory, protein_id + ".pssm")):
            pd_names = list(range(22))
            pd_names.extend(AMINO_ACIDS)
            pd_names.extend(list(range(22,25)))
            df_pssm = pd.read_csv(join(profiles_directory, protein_id + ".pssm"),
                                  delim_whitespace=True,
                                  skiprows=3, skipfooter=5,
                                  names=pd_names, engine='python',
                                  usecols=AMINO_ACIDS)
            pssm = df_pssm.to_numpy() / float(100)
            assert pssm.shape[1] == 20
        else:
            if seq is not None:
                pssm = np.zeros((len(seq),20))
            else:
                return None, flag

        if(pssm.sum(axis=None)) == 0:
            if seq is not None:
                flag = True
                for i, s in enumerate(seq):
                    if s == "X" or s == "b":
                        return None, flag
                    pssm[i, AMINO_ACIDS.index(s.upper())] = 1
                assert pssm.shape[0] == len(seq)
            else:
                return None, flag

        return pssm, flag


    @staticmethod
    def parse_list_ids(id_file):
        if '.tsv' in id_file:
            df = pd.read_csv(id_file, sep="\t")
            list_id = df["DomainID"]
        else:
            with open(id_file, 'r') as f:
                list_id = f.read()
                list_id = list_id.splitlines()
        return list_id


    @staticmethod
    def parse_fasta(protein_id, fasta_directory="./fasta/", extension=".fasta"):
        with open(join(fasta_directory, protein_id + extension), 'r') as f:
            file_content = f.read()
        file_lines = file_content.splitlines()
        proteins = []
        content = ""
        for line in file_lines:
            if '>' == line[0]:
                if len(content) > 0:
                    proteins.append((prot_id, content))
                content = ""
                prot_id = line[1:]
            else:
                assert prot_id is not None
                content += line.strip()

        if len(content) > 0:
            proteins.append((prot_id, content))

        if len(proteins) == 1:
            proteins = proteins[0]
        return proteins


    @staticmethod
    def parse_secondary_structure(protein_id, ss_folder="./dssp/", ss_type="dssp", chains=None):
        ss = ""
        if ss_type == "fasta":
            ss = Parser.parse_fasta(protein_id, fasta_directory=ss_folder, extension='.dssp')[1]
            return ss
        if ss_type == "dssp":
            if '_' in protein_id:
                protein_id = protein_id.split('_')[0]
            d = make_dssp_dict(join(ss_folder, protein_id + ".dssp"))[0]
            chain = chains[protein_id]
            seq = ""
            ss_seq = ""
            for key in d.keys():
                if key[0] != chain:
                    continue
                res = d[key][0]
                ss = d[key][1]
                if ss in ["H", "G", "I"]:
                    ss = "H"
                elif ss in ["B", "E"]:
                    ss = "E"
                elif ss in ["T", "S"]:
                    ss = "-"
                seq += res
                ss_seq += ss
            assert len(seq) == len(ss_seq)
            return ss_seq, seq
