import argparse
import numpy
import os
import random

from utils import load_comp_model_name, load_vec_files, read_full_wiki_vectors
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
parser.add_argument('--corpus', choices=['joint', 'opensubtitles', 'wikipedia'], required=True)
parser.add_argument('--model', choices=['ITBERT', 'MBERT', 'GILBERTO',
                                        'ITGPT2small', 'ITGPT2medium',
                                        'geppetto', 'xlm-roberta-large',
                                        'gpt2-xl', 'BERT_large',
                                        ],
                    required=True, help='Which model?')
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
                                        'top_two', 
                                        'top_eight', 
                                        'upper_four',
                                        'upper_six',
                                        'cotoletta_four',
                                        'cotoletta_six',
                                        'cotoletta_eight',
                                        'jat_mitchell',
                                        ],
                    required=True, help='Which layer?')
parser.add_argument('--debugging', action='store_true')
parser.add_argument(
                    '--corpus_portion',
                    choices=['entity_articles', 'full_corpus'],
                    required=True
                    )
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
args = parser.parse_args()

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder = load_vec_files(args, computational_model)

entity_vectors, all_sentences = read_full_wiki_vectors(vecs_file, out_shape)

collector = dict()

for s in range(1, 34):
    rankings = os.path.join(rankings_folder, 'sub-{:02}.ranking'.format(s))
    with open(rankings) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    rankings = {l[0] : [int(d) for d in l[1:]] for l in lines}
    for stim, ordered_idxs in rankings.items():
        #vec = [all_vecs[ranking_stim][n] for n in rankings[ranking_stim][:10]]
        #vec = numpy.average(vec, axis=0)
        #assert vec.shape == (1024, )
        #o.write('{}\t'.format(ranking_stim.replace('_', ' ')))
        #for dim in vec:
        #    o.write('{}\t'.format(dim))
        #o.write('\n')
        try:
            collector[stim].extend(ordered_idxs[:3])
            #collector[stim].extend(ordered_idxs[:3])
        except KeyError:
            #collector[stim] = ordered_idxs[:3]
            collector[stim] = ordered_idxs[:3]

random.seed(11)
max_n = 100
#max_n = 24
out_folder = os.path.join('vectors', args.corpus, args.language, args.corpus_portion, args.layer)
os.makedirs(out_folder, exist_ok=True)
collector = {k : random.sample(set(v), k=min(len(set(v)), max_n)) for k, v in collector.items()}
with open(os.path.join(out_folder, 'exp_{}_{}_{}_replication_indices.tsv'.format(args.experiment_id, computational_model, args.language)), 'w') as o:
    o.write('entity\t{}_replication_indices\n'.format(computational_model))
    for k, v in collector.items():
        o.write('{}\t'.format(k))
        for idx in v:
            o.write('{}\t'.format(idx))
        o.write('\n')
with open(os.path.join(out_folder, 'exp_{}_{}_{}_vectors.tsv'.format(args.experiment_id, computational_model, args.language)), 'w') as o:
    o.write('entity\t{}_entity_vector\n'.format(computational_model))
    for k, v in collector.items():
        vec = [entity_vectors[k][n] for n in v]
        vec = numpy.average(vec, axis=0)
        assert vec.shape == out_shape
        o.write('{}\t'.format(k))
        for dim in vec:
            o.write('{}\t'.format(dim))
        o.write('\n')
