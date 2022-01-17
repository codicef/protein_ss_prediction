# Protein Secondary Structure Prediction
Project about Protein Secondary Structure Prediction. Based on JPred training set and on a self-generated unseen blind set.

Methods and results available at https://github.com/codicef/protein_ss_prediction/blob/main/notes/codice_francesco_lab2_project.pdf

Folders content
- ./src
  - ss_models.py : file containing GOR method implementation with wrapper, SVM classifier with relative wrapper and an abstract class implementing methods for crossvalidation and performance assessment.
  - parsing_utils.py : script containing methods to parse data such as dssp, protein profiles or fasta files.
  - build_training_set.py : script to generate, given a list of ids, a training/test set for models fitting and testing.

- ./data : folder containing data for models training and testing. Data are present only in .pickle format and as protein id list. For complete RAW data contact me at francesco[dot]codice3[at]studio[dot]unibo[dot]it 
- ./notes : folder containing paper with methods and results.

