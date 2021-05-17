import torch
import numpy
import pickle
import sys
import logging
import tagme
import os
import collections
import re

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from extract_transe import entity_to_Q_id, \
                           Q_to_line, get_transe_vectors

ernie_path = '/import/cogsci/andrea/dataset/word_vectors/ERNIE'
sys.path.append(ernie_path)

from knowledge_bert import BertTokenizer, BertModel, BertForMaskedLM


def ernie(entities_and_sentences_dict, args):

    ernie_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', \
                               'word_vectors', 'ERNIE')
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(ernie_path, \
                                                        'ernie_base'))

    ent_2_q = entity_to_Q_id(ernie_path)
    q_2_line = Q_to_line(ernie_path)

    vecs = get_transe_vectors(ernie_path)
    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed)
    model, _ = BertModel.from_pretrained(os.path.join(ernie_path, \
                                                      'ernie_base'))
    model.eval()

    ernie_vectors = collections.defaultdict(list)       

    for entity, sentences in tqdm(entities_and_sentences_dict.items()):        

        if entity == 'Sagrada_Família':
            entity = 'Sagrada Família'

        try:
            entity_id = ent_2_q[entity]
        except KeyError:
            print('missing: {}'.format(entity))

        for sentence in sentences:
            ### Removing BERT's annotation
            if args.extraction_mode == 'full_sentence' or args.extraction_mode == 'unmasked':
                sentence = sentence.replace('[SEP]', ' ')
            elif args.extraction_mode == 'masked':
                sentence = re.sub('\[SEP\].+\[SEP\]', '[MASK]', sentence)
            tokens = re.sub('\s+', ' ', sentence)
            tokens, _ = tokenizer.tokenize(tokens, '')
            #input_mask = [1] * len(tokens)

            # Creating entity tokens
            tokenized_entity, _ = tokenizer.tokenize(entity, '')
            name = entity.split() + \
                   entity.lower().split() + \
                   [t for t in tokenized_entity if len(t)>1]
            relevant_indices = [t_i for t_i, t in enumerate(tokens) if t in name]

            if len(relevant_indices) >= 1:
                entities = ['UNK' if t_i != max(relevant_indices) else entity_id \
                            for t_i, t in enumerate(tokens)]

                tokens = ["[CLS]"] + tokens + ["[SEP]"] 
                ents = ["UNK"] + entities + ["UNK"]
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

                indexed_ents = []
                ent_mask = []
                for ent in ents:
                    if ent != "UNK":
                        indexed_ents.append(q_2_line[entity_id])
                        ent_mask.append(1)
                    else:
                        indexed_ents.append(-1)
                        ent_mask.append(0)
                ent_mask[0] = 1

                assert len(indexed_tokens) == len(indexed_ents)

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                ents_tensor = torch.tensor([indexed_ents])
                ents_tensor = embed(ents_tensor+1)
                #segments_tensors = torch.tensor([segments_ids])
                ent_mask = torch.tensor([ent_mask])

                # Predict all tokens
                with torch.no_grad():
                    encoded_tokens = model(input_ids=tokens_tensor, input_ent=ents_tensor, ent_mask=ent_mask)[0]
                    layers = list()
                    for layer in encoded_tokens:
                        layer = layer.detach().numpy()[0, 1:-1, :]
                        if args.extraction_mode == 'full_sentence':
                            layer = numpy.average(layer, axis=0)
                        elif args.extraction_mode == 'unmasked':
                            layer = layer[max(relevant_indices)+1, :]
                        assert layer.shape == (768, )
                        layers.append(layer)
                    layers = numpy.array(layers)
                    
                    ernie_vectors[entity].append((sentence, layers))
            else:
                import pdb; pdb.set_trace()
                print('error with sentence {}'.format(sentence))

    return ernie_vectors
'''

# Tokenized input
text_a = "Who was Jim Henson ? "
text_b = "Jim Henson"

# TAGME
# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "d66017b9-6aec-4298-a766-94bbb65053bb-843339462"
text_a_ann = tagme.annotate(text_a)
text_b_ann = tagme.annotate(text_b)

# Read entity map
ent_map = {}
with open(os.path.join(ernie_path, \
          "kg_embed/entity_map.txt")) as fin:
    for line in fin:
        name, qid = line.strip().split("\t")
        ent_map[name] = qid

def get_ents(ann):
    ents = []
    # Keep annotations with a score higher than 0.3
    for a in ann.get_annotations(0.3):
        if a.entity_title not in ent_map:
            continue
        ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
    return ents
        
ents_a = get_ents(text_a_ann)
ents_b = get_ents(text_b_ann)

# Tokenize
tokens_a, entities_a = tokenizer.tokenize(text_a, ents_a)
tokens_b, entities_b = tokenizer.tokenize(text_b, ents_b)
import pdb; pdb.set_trace()

tokens_one = ["[CLS]"] + tokens_a + ["[SEP]"] 
tokens_two = ["[CLS]"] + tokens_b + ["[SEP]"]
ents_one = ["UNK"] + entities_a + ["UNK"]
ents_two = ["UNK"] + entities_b + ["UNK"]
#ents = ["UNK"] + entities_a + ["UNK"] + entities_b + ["UNK"]
#segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
input_mask_one = [1] * len(tokens_one)
input_mask_two = [1] * len(tokens_two)

# Mask a token that we will try to predict back with `BertForMaskedLM`
#masked_index = 8
#tokens[masked_index] = '[MASK]'

# Convert token to vocabulary indices
indexed_tokens_one = tokenizer.convert_tokens_to_ids(tokens_one)
indexed_tokens_two = tokenizer.convert_tokens_to_ids(tokens_two)


# Convert ents
entity2id = {}
with open(os.path.join(ernie_path, \
          "kg_embed/entity2id.txt")) as fin:
    fin.readline()
    for line in fin:
        qid, eid = line.strip().split('\t')
        entity2id[qid] = int(eid)


vecs = []
vecs.append([0]*100)
with open(os.path.join(ernie_path, \
          "kg_embed/entity2vec.vec"), 'r') as fin:
    for line in fin:
        vec = line.strip().split('\t')
        vec = [float(x) for x in vec]
        vecs.append(vec)

embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed)

sentences_test = [[ents_one, indexed_tokens_one, 3], [ents_two, indexed_tokens_two, 1]]

test_vectors = []

for l in sentences_test:

    ents = l[0]
    indexed_tokens = l[1]
    index = l[2]

    # Load pre-trained model (weights)
    #model, _ = BertForMaskedLM.from_pretrained('ernie_base')
    model, _ = BertModel.from_pretrained(os.path.join(ernie_path, \
                                                      'ernie_base'))
    model.eval()
    indexed_ents = []
    ent_mask = []
    for ent in ents:
        if ent != "UNK" and ent in entity2id:
            indexed_ents.append(entity2id[ent])
            ent_mask.append(1)
        else:
            indexed_ents.append(-1)
            ent_mask.append(0)
    ent_mask[0] = 1

    print(indexed_tokens, indexed_ents)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    ents_tensor = torch.tensor([indexed_ents])
    #segments_tensors = torch.tensor([segments_ids])
    ent_mask = torch.tensor([ent_mask])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    ents_tensor = embed(ents_tensor+1).to('cuda')
    ent_mask = ent_mask.to("cuda")
    #segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        encoded_layers = model(input_ids=tokens_tensor, input_ent=ents_tensor, ent_mask=ent_mask)
        relevant_tensor = encoded_layers
        test_vectors.append(relevant_tensor)
'''
