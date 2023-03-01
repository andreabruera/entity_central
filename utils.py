import argparse
import numpy
import os
import re

def read_sentences_folder(args):

    out_folder = os.path.join('sentences', args.corpus, args.language, args.experiment_id)
    os.makedirs(out_folder, exist_ok=True)

    return out_folder

def return_entity_file(entity):
    entity = '{}{}'.format(entity[0].capitalize(), entity[1:])
    wiki_file = '{}.txt'.format(re.sub('[^a-zA-Z0-9]', '_', entity))
    return wiki_file

def load_comp_model_name(args):
    if args.model == 'ITGPT2medium':
        model_name = 'GroNLP/gpt2-medium-italian-embeddings'
        computational_model = 'gpt2'
        out_shape = (1024, )
    if args.model == 'gpt2-xl':
        model_name = 'gpt2-xl'
        computational_model = 'gpt2-xl'
        out_shape = (1600, )
    if args.model == 'MBERT':
        model_name = 'bert-base-multilingual-cased'
        computational_model = 'MBERT'
        out_shape = (768, )
    if args.model == 'BERT_large':
        model_name = 'bert-large-cased'
        computational_model = 'BERT_large'
        out_shape = (1024, )
    if args.model == 'xlm-roberta-large':
        model_name = 'xlm-roberta-large'
        computational_model = 'xlm-roberta-large'
        out_shape = (1024, )

    return model_name, computational_model, out_shape

def read_args():

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
    parser.add_argument('--corpus', choices=['opensubtitles', 'wikipedia', 'joint'], required=True)
    parser.add_argument('--layer', choices=[
                                            'low_four',
                                            'mid_four', 
                                            'top_four',
                                            'low_six',
                                            'mid_six', 
                                            'top_six',
                                            'top_eight',
                                            'low_twelve',
                                            'mid_twelve',
                                            'top_twelve', 
                                            'top_two', 
                                            'top_one', 
                                            'upper_four',
                                            'upper_six',
                                            'cotoletta_four',
                                            'cotoletta_six',
                                            'cotoletta_eight',
                                            'jat_mitchell',
                                            ],
                        required=True, help='Which layer?')
    parser.add_argument('--model', choices=[
                                            'BERT_large',
                                            'MBERT', 
                                            'ITGPT2medium',
                                            'xlm-roberta-large',
                                            'gpt2-xl',
                                            ],
                        required=True, help='Which model?')
    parser.add_argument('--cuda', choices=['0', '1', '2',
                                           ],
                        required=True, help='Which cuda device?')
    parser.add_argument('--debugging', action='store_true')
    args = parser.parse_args()

    return args

def read_full_wiki_vectors(vecs_file, out_shape):
    with open(vecs_file, 'r') as o:
        lines = [l.strip().split('\t') for l in o.readlines()]
    entity_vectors = dict()
    all_sentences = dict()
    marker = False
    for l in lines:
        vec = numpy.array(l[2:], dtype=numpy.float64)
        try:
            assert vec.shape == out_shape
        except AssertionError:
            marker = True
            continue
        try:
            entity_vectors[l[0]].append(vec)
            all_sentences[l[0]].append(l[1])
        except KeyError:
            entity_vectors[l[0]] = [vec]
            all_sentences[l[0]] = [l[1]]
    if marker:
        print('careful, some sentences had the wrong dimensionality!')

    return entity_vectors, all_sentences

def load_vec_files(args, computational_model):

    base_folder = os.path.join(
                              'contextualized_vector_selection', 
                              args.corpus,
                              args.language,
                              computational_model,
                              args.layer,
                              )
    vecs_folder = os.path.join(
                              base_folder,
                              'phrase_vectors',
                              args.corpus_portion,
                              )

    os.makedirs(vecs_folder, exist_ok=True)
    vecs_file= os.path.join(
                            vecs_folder, 
                            'all.vectors'
                            )
    rankings_folder = os.path.join(
                              base_folder,
                              'rankings',
                              args.corpus_portion,
                              )

    os.makedirs(rankings_folder, exist_ok=True)

    return vecs_file, rankings_folder
