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

#from utils import levenshtein, load_subject_runs, read_events, read_vectors
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead

#from utils import read_brain_mask, read_stimuli_and_cats, shorten_stimuli

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--sentences_file',
                    required=True,
                    )
parser.add_argument(
                    '--output_identifier',
                    required=True,
                    help='how to name the resulting frequencies file'
                    )
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
parser.add_argument('--model', choices=['ITBERT', 'MBERT', 'GILBERTO',
                                        'ITGPT2small', 'ITGPT2medium',
                                        'geppetto', 'xlm-roberta-large',
                                        ],
                    required=True, help='Which model?')
parser.add_argument('--cuda', choices=['0', '1', '2',
                                       ],
                    required=True, help='Which cuda device?')
''' 
parser.add_argument('--spatial_analysis', 
                    choices=[ 
                            'vmpfc',
                            'whole_brain', 
                            'fedorenko_language', 
                            'control_semantics', 
                            'general_semantics',
                            'bilateral_atls',
                            'left_atl',
                            'right_atl',
                            'pSTS',
                            'IFG',
                            ], 
                    required=True, 
                    help = 'Specifies how features are to be selected')
parser.add_argument('--contextualized_selection_method', 
                    choices=[
                             'brain', 
                             'concreteness',
                             ], 
                    required=True, 
                    help = 'Specifies how features are to be selected')
parser.add_argument('--analysis', required=True, \
                    choices=[
                             'whole_trial', 
                             'glm', 
                             'whole_trial_flattened'
                             ] + [
                             'time_resolved_{}'.format(t) for t in range(-2, 16)
                             ],
                    help='Average time points, or run classification'
                    )
parser.add_argument('--beg', 
                    choices=[
                             'all',
                             '2', 
                             '3',
                             '4'
                             ],
                    required=True,
                    help = 'When to start extracting the img')
parser.add_argument('--end', 
                    choices=[
                             'all',
                             '8', 
                             '9',
                             '10',
                             '11',
                             '12'
                             ],
                    required=True,
                    help = 'When to end extracting the img'
                         'time point by time point?')
''' 
parser.add_argument('--debugging', action='store_true')
args = parser.parse_args()

if args.model == 'ITBERT':
    model_name = 'dbmdz/bert-base-italian-xxl-cased'
if args.model == 'GILBERTO':
    model_name = 'idb-ita/gilberto-uncased-from-camembert'
if args.model == 'ITGPT2small':
    model_name = 'GroNLP/gpt2-small-italian'
if args.model == 'ITGPT2medium':
    model_name = 'GroNLP/gpt2-medium-italian-embeddings'
    computational_model = 'gpt2'
if args.model == 'geppetto':
    model_name = 'LorenzoDeMattei/GePpeTto'
if args.model == 'MBERT':
    model_name = 'bert-base-multilingual-cased'
if args.model == 'xlm-roberta-large':
    model_name = 'xlm-roberta-large'
    computational_model = 'xlm-roberta-large'

vecs_folder = os.path.join(
                          'contextualized_vector_selection', 
                          args.output_identifier,
                          computational_model,
                          'phrase_vectors',
                          args.layer,
                          )

os.makedirs(vecs_folder, exist_ok=True)
vecs_file= os.path.join(
                        vecs_folder, 
                        'all_{}_entity_vectors.vector'.format(computational_model)
                        )
### extracting them if not available
all_sentences = dict()
with open(args.sentences_file) as i:
    with tqdm() as counter:
        for l in i:
            name_and_sent = l.strip().split('\t')
            try:
                all_sentences[name_and_sent[0]].append(name_and_sent[1])
                counter.update(1)
            except KeyError:
                all_sentences[name_and_sent[0]] = [name_and_sent[1]]

all_sentences = {k : random.sample(v, k=min(1000, len(v))) for k, v in all_sentences.items()}

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
            if args.debugging:
                if l_i >= 100:
                    continue
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
                    #print(l)
                    entity_vectors[stimulus].append((numpy.zeros(shape=(1024,)), old_l))
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
                    entity_vectors[stimulus].append((numpy.zeros(shape=(1024,)), old_l))
                    continue
                try:
                    inputs = tokenizer(l, return_tensors="pt").to(cuda_device)
                except RuntimeError:
                    print('input error')
                    print(l)
                    entity_vectors[stimulus].append((numpy.zeros(shape=(1024,)), old_l))
                    continue
                try:
                    outputs = model(**inputs, output_attentions=False, \
                                    output_hidden_states=True, return_dict=True)
                except RuntimeError:
                    print('output error')
                    print(l)
                    entity_vectors[stimulus].append((numpy.zeros(shape=(1024,)), old_l))
                    continue

                hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                if len(split_spans) > 1:
                    split_spans = [split_spans[-1]]
                for beg, end in split_spans:
                    if len(tokenizer.tokenize(l)[beg:end]) == 0:
                        entity_vectors[stimulus].append((numpy.zeros(shape=(1024,)), old_l))
                        continue
                    print(tokenizer.tokenize(l)[beg:end])
                    ### If there are less than two tokens that must be a mistake
                    if len(tokenizer.tokenize(l)[beg:end]) < 2:
                        entity_vectors[stimulus].append((numpy.zeros(shape=(1024,)), old_l))
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
            assert vec.shape == (1024, )
            line = line.replace('\t', ' ')
            o.write('{}\t{}\t'.format(stim, line))
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')
