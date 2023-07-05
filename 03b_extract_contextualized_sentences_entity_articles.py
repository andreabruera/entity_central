import argparse
import itertools
import joblib
import nilearn
import random
import numpy
import os
import re
import scipy
import shutil
import spacy
import sklearn
import torch

from joblib import parallel_backend
from nilearn import image, masking
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead

from utils import layers, load_comp_model_name, load_vec_files, read_args, read_brain_data, read_entity_sentences, read_full_wiki_vectors, read_sentences_folder
from exp_two_utils import read_personally_familiar_sentences

def sensible_finder(all_args):

    lines = all_args[0]
    stimuli = all_args[1]
    args = all_args[2]
    current_stimulus = all_args[3]

    ent_sentences = {s : list() for s in stimuli}

    with tqdm() as counter:
        for l in lines:
            ent_sentences[stimulus].append(new_l)
            counter.update(1)

    return ent_sentences

args = read_args(vector_extraction=True, contextualized_selection=True)
if args.experiment_id == 'two' and args.corpus_portion != 'entity_sentences':
    raise(RuntimeError)
brain_data = read_brain_data(args)

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder, repl_file = load_vec_files(args, computational_model)
sent_len_threshold = 20 

original_sentences = read_entity_sentences(args)
all_sentences = {k : list() for k in original_sentences.keys()}
all_lengths = dict()
for k, sents in original_sentences.items():
    clean_sents = list()
    current_sent = ''
    for line in sents:
        if len(line.split()) > sent_len_threshold:
            clean_sents.append(line)
        else:
            current_sent = '{}, {}'.format(current_sent, line)
            if len(current_sent.split()) > sent_len_threshold:
                clean_sents.append(current_sent)
                current_sent = ''
    if current_sent != '':
        clean_sents.append(current_sent)
    all_sentences[k] = clean_sents

### lengths
lengths = [len([tok for sent in v for tok in sent.split()]) for v in all_sentences.values()]
# min -1., max -.1
for sub in range(1, 34):
    pers_lengths = {'{:02}_{}'.format(sub, k) : ((1-.1)*(((len([tok for sent in v for tok in sent.split()])-min(lengths))/(max(lengths)-min(lengths)))))-1. for k, v in all_sentences.items()}
    #pers_lengths = {'{:02}_{}'.format(sub, k) : (2.*(((len([tok for sent in v for tok in sent.split()])-min(lengths))/(max(lengths)-min(lengths)))))-1. for k, v in all_sentences.items()}
    all_lengths.update(pers_lengths)

if args.corpus_portion == 'entity_sentences':
    if args.language == 'it':
        spacy_model = spacy.load("it_core_news_lg")
    if args.language == 'en':
        spacy_model = spacy.load("en_core_web_lg")
    relevant_pos = [
               'ADJ', 
               'ADV', 
               'NOUN', 
               'PROPN', 
               'VERB',
               #'X',
               'NUM',
               'AUX',
               ]
    to_be_ignored = [
                     'OTHER_NAME', 
                     'STREET_NAME', 
                     'PERSON_NAME', 
                     'PROPER_NAME', 
                     'PEOPLE_NAME'
                     ]

if args.experiment_id == 'two':
    formula_one = '(?<=\[\[)(.+?)\|.+?(?=\]\])'
    formula_two = '(?<=\[\[)(.+?)(?=\]\])'
    pers_sentences = read_personally_familiar_sentences()
    for sub, sub_dict in pers_sentences.items():
        ### lengths
        lengths = [len([tok for sent in v for tok in sent.split()]) for v in sub_dict.values()]
        # min .1, max +1
        pers_lengths = {'{:02}_{}'.format(sub, k) : ((1.-.1)*(((len([tok for sent in v for tok in sent.split()])-min(lengths))/(max(lengths)-min(lengths)))))+.1 for k, v in sub_dict.items()}
        #pers_lengths = {'{:02}_{}'.format(sub, k) : (2.*(((len([tok for sent in v for tok in sent.split()])-min(lengths))/(max(lengths)-min(lengths)))))-1. for k, v in sub_dict.items()}
        for k in pers_lengths.keys():
            assert k not in all_lengths.keys()
        all_lengths.update(pers_lengths)
        for name, sents in sub_dict.items():
            clean_sents = list()
            current_sent = ''
            for line in sents:
                line = re.sub('_', ' ', line)
                line = re.sub(formula_one, r'\1', line)
                matches = re.findall(formula_two, line)
                #replacements = [re.sub('[^A-Za-z0-9]', '_', match) for match in matches]
                for original in matches:
                    line = line.replace('[[{}]]'.format(original), original)
                #line = re.sub('[^A-Za-z0-9_ ]', ' ', line).lower().split()
                line = re.sub('[^A-Za-z0-9_ ]', ' ', line)
                to_be_ignored = ['OTHER_NAME', 'STREET_NAME', 'PERSON_NAME', 'PROPER_NAME', 'PEOPLE_NAME']
                for ig in to_be_ignored:
                    line = line.replace(ig, '')
                #line = line.lower()
                ### putting words together, if the sentence is too short, 
                ### to help contextualized models to have actual context 
                ### 20 was very good
                #sent_len_threshold = 20
                if len(line.split()) > sent_len_threshold:
                    clean_sents.append(line)
                else:
                    current_sent = '{}, {}'.format(current_sent, line)
                    if len(current_sent.split()) > sent_len_threshold:
                        clean_sents.append(current_sent)
                        current_sent = ''
            if current_sent != '':
                clean_sents.append(current_sent)

            all_sentences['{:02}_{}'.format(sub, name)] = clean_sents

### writing to file the vectors

lengths_file = os.path.join(
                        'static_vectors', 
                        'exp_{}_sentence_lengths_{}_ratings.tsv'.format(
                                   args.experiment_id,
                                   args.language, 
                                   )
                        )
with open(lengths_file, 'w') as o:
    for stim, val in all_lengths.items():
        o.write('{}\t{}\t{}\n'.format(stim.split('_')[0], stim.split('_')[1], val))
#import pdb; pdb.set_trace()
cuda_device = 'cuda:{}'.format(args.cuda)

if 'GeP' in model_name:
    model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto").to(cuda_device)
    required_shape = model.config.n_embd
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
elif 'gpt' in model_name or 'GPT' in model_name:
    model = AutoModel.from_pretrained(model_name).to(cuda_device)
    required_shape = model.embed_dim
    print(required_shape)
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(cuda_device)
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers
tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    sep_token='[SEP]', 
                    #max_length=None, 
                    max_length=max_len,
                    truncation=True
                    )

print('Dimensionality: {}'.format(required_shape))
print('Number of layers: {}'.format(n_layers))

entity_vectors = dict()
layer_start, layer_end = layers(args, n_layers)

### encoding all sentences
with tqdm() as pbar:
    for stimulus, stim_sentences in all_sentences.items():
        counter = 0
        entity_vectors[stimulus] = list()
        assert len(stim_sentences) >= 1
        for l_i, l in enumerate(stim_sentences):
            if len(l.strip()) == 0:
                continue

            if args.debugging:
                counter += 1
                if counter > 10:
                    continue

            ### finding out where the context words are

            ### finding out the words
            spacy_line = spacy_model(l)
            new_line = ''
            ### words
            for tok in spacy_line:
                if len(tok.text) < 3:
                    new_line = '{} {}'.format(new_line, tok.text)
                elif (tok.pos_ not in relevant_pos and '_' not in tok.text) or tok.text in to_be_ignored:
                    #print((tok.pos_, tok.text))
                    new_line = '{} {}'.format(new_line, tok.text)
                    #continue
                else:
                    new_line = '{} [SEP] {} [SEP]'.format(new_line, tok.text)
            new_inputs = tokenizer(new_line, return_tensors="pt")
            spans = [i_i for i_i, i in enumerate(new_inputs['input_ids'].numpy().reshape(-1)) if 
                    i==tokenizer.convert_tokens_to_ids(['[SEP]'])[0]]
            if 'bert' in model_name and len(spans)%2==1:
                spans = spans[:-1]
            #if 'bert' in model_name:
            #    spans = spans[1:-1]
            assert len(spans) % 2 == 0

            spacy_txt = ' '.join(['{}'.format(tok.text) for tok in spacy_line])

            if len(spans) > 1:
                tkz = list()
                corr_spans = list()
                for tok in new_inputs['input_ids'][0]:
                    if tok == tokenizer.convert_tokens_to_ids(['[SEP]'])[0]:
                        corr_spans.append(len(tkz))
                    else:
                        tkz.append(tok)
                split_spans = list()
                for i in list(range(len(corr_spans)))[::2]:
                    current_span = (corr_spans[i], corr_spans[i+1])
                    split_spans.append(current_span)

                if len(split_spans) > 0:
            
                    #old_inputs = tokenizer(spacy_txt, return_tensors="pt")

                    ### tokenizing inputs
                    inputs = tokenizer(
                                       spacy_txt, 
                                       return_tensors="pt", 
                                       truncation=True
                                       ).to(cuda_device)

                    ### checking words
                    words = [tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().detach()[beg:end]) for beg, end in split_spans]
                    print(words)

                    ### encoding words
                    outputs = model(
                                    **inputs, 
                                    output_attentions=False,
                                    output_hidden_states=True, 
                                    return_dict=True
                                    )

                    hidden_states = numpy.array(
                                       [s[0].cpu().detach().numpy() for s in outputs['hidden_states']]
                                           )
                    '''
                    if '_bert' in args.model.lower():
                        beg = 1
                    else:
                        beg = 0
                    if len(re.findall('\W$', l)) > 0:
                        end = -1
                    else:
                        end = len(tokenizer.tokenize(l))
                    print(tokenizer.tokenize(l)[beg:end])
                    ### If there are less than two tokens that must be a mistake
                    if len(tokenizer.tokenize(l)[beg:end]) < 1:
                        #entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), l))
                        print(len(tokenizer.tokenize(l)))
                        print('\nERROR {}\n'.format(l))
                        print(tokenizer.tokenize(l)[beg:end])
                        continue
                    mention = hidden_states[:, beg:end, :]
                    mention = numpy.average(mention, axis=1)
                    mention = mention[layer_start:layer_end, :]

                    mention = numpy.average(mention, axis=0)
                    assert mention.shape == (required_shape, )
                    entity_vectors[stimulus].append((mention, l))
                    pbar.update(1)
                    '''
                    sentence_vectors = list()
                    for beg, end in split_spans:
                        if len(inputs['input_ids'][0][beg:end].cpu().detach()) < 1:
                            print('\n\nERROR {}\n\n'.format(l))
                            #import pdb; pdb.set_trace()
                            continue
                        ### spans
                        mention = hidden_states[:, beg:end, :]
                        mention = numpy.nanmean(mention, axis=1)
                        ### layers
                        mention = mention[layer_start:layer_end, :]
                        mention = numpy.nanmean(mention, axis=0)
                        assert mention.shape == (required_shape, )
                        sentence_vectors.append(mention)
                    print(
                          'used {} tokens, against a total of {}'.format(
                                    len(sentence_vectors), 
                                    len(spacy_txt.split())
                                    )
                          )
                    entity_vectors[stimulus].append((numpy.nanmean(sentence_vectors, axis=0), l))
                    pbar.update(1)


with open(vecs_file, 'w') as o:
    for stim, vecz in entity_vectors.items():
        for vec, line in vecz:
            try:
                assert vec.shape == out_shape
            except AssertionError:
                print([stim, line])
            line = line.replace('\t', ' ')
            o.write('{}\t{}\t'.format(stim, line))
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')

### writing to file the vectors

with open(repl_file, 'w') as o:
    o.write('subject\tentity\tvector\n')
    for s, sub_data in tqdm(brain_data.items()):

        current_entity_vectors = {k : v for k, v in entity_vectors.items() if k in sub_data.keys()}
        for k, v in entity_vectors.items():
            if '{:02}_'.format(s) in k:
                current_entity_vectors[k.replace('{:02}_'.format(s), '')] = v
        for stim, vecz in current_entity_vectors.items():
            print([s, len(vecz)])
            vec = numpy.nanmean([v[0] for v in vecz], axis=0)
            #for vec, line in vecz:
            try:
                assert vec.shape == out_shape
            except AssertionError:
                print([stim, line])
            #line = line.replace('\t', ' ')
            o.write('{}\t{}\t'.format(s, stim))
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')
