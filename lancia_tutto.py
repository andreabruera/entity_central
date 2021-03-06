import os

extraction_methods = ['masked', 'unmasked']

for e in extraction_methods:
    os.system('python3 main.py \
              --word_selection eeg_one \
              --model bert \
              --output_folder /import/cogsci/andrea/github/fame/word_vectors \
              --extraction_mode {}'.format(e)) 
