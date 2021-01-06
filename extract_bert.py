import os
import re
import collections
import torch
import numpy
import argparse
import nltk

from transformers import BertModel, BertTokenizer
from tqdm import tqdm

def vector_to_txt(word, vector, output_file):
    output_file.write('{}\t'.format(word))
    for dimension_index, dimension_value in enumerate(vector):
        if dimension_index != len(vector)-1:
            output_file.write('{}\t'.format(dimension_value))
        else: 
            output_file.write('{}\n'.format(dimension_value))

# Create multiple vectors for Bert clustering analysis

def bert(entities_and_sentences_dict, out_folder='/import/cogsci/andrea/dataset/word_vectors/bert_4_layers'):

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased')

    bert_vectors = collections.defaultdict(list)       

    ### Extracting the BERT vectors
    for extraction_method in ['unmasked', 'full_sentence', 'masked']:
        print('Now extracting the vectors in modality {}'.format(extraction_method))
        extraction_method_folder = os.path.join(out_folder, extraction_method)
        os.makedirs(extraction_method_folder, exist_ok=True)

        for entity, sentences in tqdm(entities_and_sentences_dict.items()):        

            ### Ready to extract the vectors

            for sentence in sentences:

                if extraction_method == 'masked':
                    sentence = re.sub('\[SEP\].+\[SEP\]', '[MASK]', sentence)
                elif extraction_method == 'full_sentence':
                    sentence = re.sub('\[SEP\] ', '', sentence)

                input_ids = bert_tokenizer(sentence, return_tensors='pt')
                readable_input_ids = input_ids['input_ids'][0].tolist()

                if len(readable_input_ids) <= 512:

                    if extraction_method == 'unmasked':
                        ### Finding the relevant indices within the '[SEP]' token

                        sep_indices = list()
                        for index, bert_id in enumerate(readable_input_ids):
                            if bert_id == 102 and index != len(readable_input_ids)-1:
                                sep_indices.append(index)
                        assert len(sep_indices)%2 == 0

                        relevant_indices = list()
                        relevant_ids = list()

                        for sep_start in range(0, len(sep_indices), 2):
                            new_window = [k-sep_start for k in range(sep_indices[sep_start], sep_indices[sep_start+1]-1)] 
                            relevant_indices.append(new_window)

                            sep_window = [k for k in range(sep_indices[sep_start]+1, sep_indices[sep_start+1])] 
                            relevant_ids.append([readable_input_ids[k] for k in sep_window])

                        input_ids = bert_tokenizer(re.sub('\[SEP\]', '', sentence), return_tensors='pt')
                        readable_input_ids = input_ids['input_ids'][0].tolist()
                        for k_index, k in enumerate(relevant_indices):
                            try:
                                assert [readable_input_ids[i] for i in k] == relevant_ids[k_index]
                                assert 102 not in [readable_input_ids[i] for i in k]
                                assert 102 not in relevant_ids[k_index]
                            except AssertionError:
                                print('There was a problem with this sentence: {}'.format(sentence))

                    elif extraction_method == 'masked':
                        relevant_indices = [[i] for i, input_id in enumerate(readable_input_ids) if input_id == 103]

                    elif extraction_method == 'full_sentence':
                        relevant_indices = [[i for i in range(1, len(readable_input_ids)-1)]]
                   
                    outputs = bert_model(**input_ids, return_dict=True, output_hidden_states=True, output_attentions=False)

                    assert len(readable_input_ids) == len(outputs['hidden_states'][1][0])
                    assert len(relevant_indices) >= 1
                    word_layers = list()

                    ### Using the first 4 layers in BERT
                    for layer in range(1, 5):
                        layer_container = list()
                        for relevant_index_list in relevant_indices:
                            for individual_index in relevant_index_list:
                                layer_container.append(outputs['hidden_states'][layer][0][individual_index].detach().numpy())
                        layer_container = numpy.average(layer_container, axis=0)
                        assert len(layer_container) == 768
                        word_layers.append(layer_container)
                    sentence_vector = numpy.average(word_layers, axis=0)
                    assert len(sentence_vector) == 768
                    bert_vectors[entity].append((sentence, sentence_vector))
                else:
                    print(sentence)

            for entity, vector_tuples in bert_vectors.items():

                with open(os.path.join(extraction_method_folder, '{}.vec'.format(re.sub(' ', '_', entity))), 'w') as o:
                    for sentence, vector in vector_tuples:
                        vector_to_txt(sentence, vector, o)
