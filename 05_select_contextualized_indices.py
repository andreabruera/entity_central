import argparse
import numpy
import os
import random

from utils import load_comp_model_name, load_vec_files, read_full_wiki_vectors
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
parser.add_argument('--corpus', choices=['opensubtitles', 'wikipedia'], required=True)
parser.add_argument('--model', choices=['ITBERT', 'MBERT', 'GILBERTO',
                                        'ITGPT2small', 'ITGPT2medium',
                                        'geppetto', 'xlm-roberta-large',
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
                                        'upper_four',
                                        'upper_six',
                                        'cotoletta_four',
                                        'cotoletta_six',
                                        'cotoletta_eight',
                                        'jat_mitchell',
                                        ],
                    required=True, help='Which layer?')
parser.add_argument('--debugging', action='store_true')
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
            #collector[stim].extend(ordered_idxs[:2])
            collector[stim].extend(ordered_idxs[:3])
        except KeyError:
            collector[stim] = ordered_idxs[:3]
            #collector[stim] = ordered_idxs[:2]

random.seed(11)
max_n = 100
#max_n = 24
os.makedirs('vectors', exist_ok=True)
collector = {k : random.sample(set(v), k=min(len(set(v)), max_n)) for k, v in collector.items()}
with open(os.path.join('vectors', 'exp_{}_{}_{}_replication_indices.tsv'.format(args.experiment_id, computational_model, args.corpus)), 'w') as o:
    o.write('entity\t{}_replication_indices\n'.format(computational_model))
    for k, v in collector.items():
        o.write('{}\t'.format(k))
        for idx in v:
            o.write('{}\t'.format(idx))
        o.write('\n')
with open(os.path.join('vectors', 'exp_{}_{}_{}_vectors.tsv'.format(args.experiment_id, computational_model, args.corpus)), 'w') as o:
    o.write('entity\t{}_entity_vector\n'.format(computational_model))
    for k, v in collector.items():
        vec = [entity_vectors[k][n] for n in v]
        vec = numpy.average(vec, axis=0)
        assert vec.shape == out_shape
        o.write('{}\t'.format(k))
        for dim in vec:
            o.write('{}\t'.format(dim))
        o.write('\n')
