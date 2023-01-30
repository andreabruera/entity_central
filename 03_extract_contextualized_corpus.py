import argparse
import itertools
import joblib
import nilearn
import random
import numpy
import os
import re
import scipy
import sklearn
import torch

from joblib import parallel_backend
from nilearn import image, masking
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead

from utils import load_comp_model_name, load_vec_files, read_full_wiki_vectors, read_sentences_folder

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
parser.add_argument(
                    '--corpus_portion',
                    choices=['entity_articles', 'full_corpus'],
                    required=True
                    )
parser.add_argument('--corpus', choices=['opensubtitles', 'wikipedia'], required=True)
parser.add_argument('--layer', choices=[
                                        'low_four',
                                        'mid_four', 
                                        'top_four',
                                        'low_six',
                                        'mid_six', 
                                        'top_six',
                                        'low_twelve',
                                        'mid_twelve',
                                        'top_twelve', 
                                        'upper_four',
                                        'upper_six',
                                        'cotoletta_four',
                                        'cotoletta_six',
                                        'cotoletta_eight',
                                        'jat_mitchell',
                                        ],
                    required=True, help='Which layer?')
parser.add_argument('--model', choices=[
                                        'MBERT', 
                                        'ITGPT2medium',
                                        'xlm-roberta-large',
                                        ],
                    required=True, help='Which model?')
parser.add_argument('--cuda', choices=['0', '1', '2',
                                       ],
                    required=True, help='Which cuda device?')
parser.add_argument('--debugging', action='store_true')
args = parser.parse_args()

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder = load_vec_files(args, computational_model)

### extracting them if not available
all_sentences = dict()
sent_folder = read_sentences_folder(args)
sent_file = os.path.join(sent_folder,'{}.sentences'.format(args.corpus_portion)) 
with open(sent_file) as i:
    with tqdm() as counter:
        for l in i:
            name_and_sent = l.strip().split('\t')
            try:
                all_sentences[name_and_sent[0]].append(name_and_sent[1])
                counter.update(1)
            except KeyError:
                all_sentences[name_and_sent[0]] = [name_and_sent[1]]

if args.debugging:
    max_n = 500
else:
    max_n = 1000
all_sentences = {k : random.sample(v, k=min(max_n, len(v))) for k, v in all_sentences.items()}

cuda_device = 'cuda:{}'.format(args.cuda)

tokenizer = AutoTokenizer.from_pretrained(model_name, sep_token='[SEP]')
if 'GeP' in model_name:
    model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto").to(cuda_device)
    required_shape = model.config.n_embd
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
elif 'gpt' in model_name or 'GPT' in model_name:
    model = AutoModel.from_pretrained(model_name).to(cuda_device)
    required_shape = model.embed_dim
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(cuda_device)
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers

print('Dimensionality: {}'.format(required_shape))
print('Number of layers: {}'.format(n_layers))

max_len = max_len - 10
entity_vectors = dict()

### encoding all sentences
with tqdm() as pbar:
    for stimulus, stim_sentences in all_sentences.items():
        entity_vectors[stimulus] = list()
        assert len(stim_sentences) >= 1
        for l_i, l in enumerate(stim_sentences):
            inputs = tokenizer(l, return_tensors="pt")

            spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                    i==tokenizer.convert_tokens_to_ids(['[SEP]'])[0]]
            if 'bert' in model_name and len(spans)%2==1:
                spans = spans[:-1]
            if len(spans) > 1:
                old_l = '{}'.format(l)
                try:
                    assert len(spans) % 2 == 0
                except AssertionError:
                    entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), old_l))
                    continue
                l = re.sub(r'\[SEP\]', '', l)
                ### Correcting spans
                correction = list(range(1, len(spans)+1))
                spans = [max(0, s-c) for s,c in zip(spans, correction)]
                split_spans = list()
                for i in list(range(len(spans)))[::2]:
                    if len(l.split()) > 5 and 'GPT' in args.model:
                        current_span = (spans[i]+1, spans[i+1])
                    else:
                        current_span = (spans[i], spans[i+1])
                    split_spans.append(current_span)

                if len(tokenizer.tokenize(l)) > max_len:
                    entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), old_l))
                    continue
                try:
                    inputs = tokenizer(l, return_tensors="pt").to(cuda_device)
                except RuntimeError:
                    print('input error')
                    print(l)
                    entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), old_l))
                    continue
                try:
                    outputs = model(**inputs, output_attentions=False, \
                                    output_hidden_states=True, return_dict=True)
                except RuntimeError:
                    print('output error')
                    print(l)
                    entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), old_l))
                    continue

                hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                if len(split_spans) > 1:
                    split_spans = [split_spans[-1]]
                for beg, end in split_spans:
                    if len(tokenizer.tokenize(l)[beg:end]) == 0:
                        entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), old_l))
                        continue
                    print(tokenizer.tokenize(l)[beg:end])
                    ### If there are less than two tokens that must be a mistake
                    if len(tokenizer.tokenize(l)[beg:end]) < 1:
                        entity_vectors[stimulus].append((numpy.zeros(shape=out_shape), old_l))
                        print('\nERROR {}\n'.format(l))
                        continue
                    mention = hidden_states[:, beg:end, :]
                    mention = numpy.average(mention, axis=1)
                    if args.layer == 'low_four':
                        layer_start = 1
                        ### outputs has at dimension 0 the final output
                        layer_end = 5
                    if args.layer == 'low_six':
                        layer_start = 1
                        ### outputs has at dimension 0 the final output
                        layer_end = 7
                    if args.layer == 'low_twelve':
                        layer_start = 1
                        ### outputs has at dimension 0 the final output
                        layer_end = 13
                    if args.layer == 'mid_four':
                        layer_start = int(n_layers/2)-2
                        layer_end = int(n_layers/2)+3
                    if args.layer == 'mid_six':
                        layer_start = int(n_layers/2)-3
                        layer_end = int(n_layers/2)+4
                    if args.layer == 'mid_twelve':
                        layer_start = int(n_layers/2)-6
                        layer_end = int(n_layers/2)+7
                    if args.layer == 'upper_four':
                        layer_start = int(n_layers/2)
                        layer_end = int(n_layers/2)+5
                    if args.layer == 'upper_six':
                        layer_start = int(n_layers/2)
                        layer_end = int(n_layers/2)+7
                    if args.layer == 'cotoletta_eight':
                        layer_start = int(n_layers/2)+int(int(n_layers/2)/2)-4
                        layer_end = int(n_layers/2)+int(int(n_layers/2)/2)+5
                    if args.layer == 'cotoletta_six':
                        layer_start = int(n_layers/2)+int(int(n_layers/2)/2)-3
                        layer_end = int(n_layers/2)+int(int(n_layers/2)/2)+4
                    if args.layer == 'cotoletta_four':
                        layer_start = int(n_layers/2)+int(int(n_layers/2)/2)-2
                        layer_end = int(n_layers/2)+int(int(n_layers/2)/2)+3
                    if args.layer == 'top_four':
                        layer_start = -4
                        ### outputs has at dimension 0 the final output
                        layer_end = n_layers+1
                    if args.layer == 'top_six':
                        layer_start = -6
                        ### outputs has at dimension 0 the final output
                        layer_end = n_layers+1
                    if args.layer == 'top_twelve':
                        layer_start = -12
                        ### outputs has at dimension 0 the final output
                        layer_end = n_layers+1
                    if args.layer == 'jat_mitchell':
                        layer_start = 10
                        layer_end = 22
                    mention = mention[layer_start:layer_end, :]

                    mention = numpy.average(mention, axis=0)
                    assert mention.shape == (required_shape, )
                    entity_vectors[stimulus].append((mention, old_l))
                    pbar.update(1)

### writing to file the vectors

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
