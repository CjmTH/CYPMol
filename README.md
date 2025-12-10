# CYPMol
A single model framework integrating Functional Residues with Protein Features and Molecule Embeddings to predict CYP Substrates, Inhibitors, and Metabolism Sites.
<img width="3560" height="3939" alt="model" src="https://github.com/user-attachments/assets/9c633575-079f-4bdf-a600-5b2ff34c46dc" />

## Setup
The environment for CYPMol is configured identically to DeepP450. For detailed setup instructions, please refer to the DeepP450 repository: https://github.com/CjmTH/DeepP450.

## training 
The raw data and training scripts for the Substrate, Inhibitor, and BoM tasks are located in the main/substrate/, main/inhibitor/, and main/BoM/ directories, respectively.

## testing
The input file formats required for CYPMol across different tasks can be found in the corresponding example.csv files within each task directory. The model weight files for each task are provided as follows:

Substrate: https://huggingface.co/CJM1111/CYPMOl/tree/main/Substrate

Inhibitor: https://huggingface.co/CJM1111/CYPMOl/tree/main/Inhibitor

BoM: https://huggingface.co/CJM1111/CYPMOl/tree/main/BoM
