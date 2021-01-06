import torch
import numpy

from tqdm import tqdm
from collections import defaultdict
from transformers import BertModel, BertTokenizer

def vector_to_txt(word, vector, output_file):
    output_file.write('{}\t'.format(word))
    for dimension_index, dimension_value in enumerate(vector):
        if dimension_index != len(vector)-1:
            output_file.write('{}\t'.format(dimension_value))
        else: 
            output_file.write('{}\n'.format(dimension_value))

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)

with open('/import/cogsci/andrea/github/fame/resources/eeg_data_ids.txt') as ids_txt:
    raw_lines = [l.strip().split('\t') for l in ids_txt.readlines()]
    lines = [l for l in raw_lines if len(l) > 1]
    words = [l[1] for l in lines]

for w in words:
    initials = w[:2]
    with open('/import/cogsci/andrea/dataset/wikipedia_article_by_article/{}/{}.txt'.format(initials, w)) as input_file:
        bert_txt = ' '.join([w for l in input_file.readlines() for w in l.split()][:129])
    tokenized_input = bert_tokenizer.encode(bert_txt)
    input_ids = torch.tensor([tokenized_input])
    with torch.no_grad():
        output = bert_model(input_ids)
    final_vector = output[2][12][0][0].numpy()
    with open('/import/cogsci/andrea/github/fame/word_vectors/bert_eeg_stanford.vec', 'a') as o:
        vector_to_txt(w, final_vector, o)
