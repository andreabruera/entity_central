import numpy
import os
import re

def return_entity_file(entity):
    entity = '{}{}'.format(entity[0].capitalize(), entity[1:])
    wiki_file = '{}.txt'.format(re.sub('[^a-zA-Z0-9]', '_', entity))
    return wiki_file

def load_comp_model_name(args):
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

    return model_name, computational_model

def read_full_wiki_vectors(vecs_file):
    with open(vecs_file, 'r') as o:
        lines = [l.strip().split('\t') for l in o.readlines()]
    entity_vectors = dict()
    all_sentences = dict()
    for l in lines:
        vec = numpy.array(l[2:], dtype=numpy.float64)
        assert vec.shape == (1024,)
        try:
            entity_vectors[l[0]].append(vec)
            all_sentences[l[0]].append(l[1])
        except KeyError:
            entity_vectors[l[0]] = [vec]
            all_sentences[l[0]] = [l[1]]
    return entity_vectors, all_sentences

def load_vec_files(args, computational_model):

    vecs_folder = os.path.join(
                              'contextualized_vector_selection', 
                              'eeg_entities_full_wiki',
                              computational_model,
                              'phrase_vectors',
                              args.layer,
                              )

    os.makedirs(vecs_folder, exist_ok=True)
    vecs_file= os.path.join(
                            vecs_folder, 
                            'all_{}_entity_vectors.vector'.format(computational_model)
                            )
    rankings_folder = vecs_folder.replace(
                              'phrase_vectors',
                              'contextualized_rankings',
                              ).replace(
                              'eeg_entities_full_wiki',
                              'eeg_entities_full_wiki_exp_{}'.format(args.experiment_id),
                              )
    os.makedirs(rankings_folder, exist_ok=True)

    return vecs_file, rankings_folder
