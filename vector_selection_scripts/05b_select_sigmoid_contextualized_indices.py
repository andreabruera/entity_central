import argparse
import matplotlib
import numpy
import os
import random
import scipy

from matplotlib import pyplot
from scipy import stats
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import load_comp_model_name, load_vec_files, read_args, read_full_wiki_vectors

def zero_one_norm(values):
    values = [(v - min(values))/(max(values)-min(values)) for v in values]
    return values

args = read_args(contextualized_selection=True)

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder = load_vec_files(args, computational_model)

entity_vectors, all_sentences = read_full_wiki_vectors(vecs_file, out_shape)

collector = {k : {s : list() for s in range(1, 34)} for k in entity_vectors.keys()}

for s in range(1, 34):
    rankings = os.path.join(rankings_folder, 'sub-{:02}.ranking'.format(s))
    if args.individuals:
        rankings = rankings.replace('.ranking', '_individuals.ranking')
    if args.predicted:
        rankings = rankings.replace('.ranking', '_model.ranking')
    #print(rankings)
    assert os.path.exists(rankings)
    with open(rankings) as i:
        lines = [l.strip().split('\t') for l in i.readlines()]
    rankings = {l[0] : [float(d.split(',')[1]) for d in l[1:]] for l in lines}
    ids = {l[0] : [int(d.split(',')[0]) for d in l[1:]] for l in lines}
    for stim, ordered_idxs in rankings.items():
        #vec = [entity_vectors[k][n] for n in data]
        #coeffs = scipy.special.logit(zero_one_norm([k for k in range(len(ordered_idxs))]))
        coeffs = scipy.special.logit(zero_one_norm(ordered_idxs))
        try:
            coeffs[0] = -int(coeffs[1]) - 1
        except OverflowError:
            coeffs[0] = -int(coeffs[2]) - 1
            coeffs[1] = -int(coeffs[2]) - 1

        coeffs[-1] = int(coeffs[-2]) + 1
        coeffs = zero_one_norm(coeffs)
        vecs = [vec[1] for k in ids[stim] for vec in entity_vectors[stim] if vec[0]==k]
        vec = numpy.average([vec*(1-coeff) for vec, coeff in zip(vecs, coeffs)], axis=0)

        #vec = [vec[1] for vec in entity_vectors[k] if vec[0] in data]
        #vec = numpy.average(vec, axis=0)
        assert vec.shape == out_shape
        #collector[stim][s].extend(random.sample(ordered_idxs[:10], k=3))
        #n_items = int(len(ordered_idxs)/10)
        #print([stim, n_items])
        collector[stim][s] = vec

random.seed(11)
#max_n = 100
#max_n = 24
out_folder = os.path.join('vectors', args.corpus, args.language, args.corpus_portion, args.layer)
os.makedirs(out_folder, exist_ok=True)
#collector = {k : random.sample(set(v), k=min(len(set(v)), max_n)) for k, v in collector.items()}
#with open(os.path.join(out_folder, 'exp_{}_{}_dirty_{}_replication_indices.tsv'.format(args.experiment_id, computational_model, args.language)), 'w') as o:
#    o.write('subject\tentity\t{}_replication_indices\n'.format(computational_model))
#    for k, v in collector.items():
#        for i in range(33):
#            sub = i+1
#            o.write('{}\t{}\t'.format(sub, k))
#            for idx in v[i]:
#                o.write('{}\t'.format(idx))
#            o.write('\n')

final_collector = {k : list() for k in collector.keys()}

out_file = os.path.join(
                        out_folder, 
                        'exp_{}_{}_gold_{}_{}_average_{}_{}_logit_vectors.tsv'.format(
                                   args.experiment_id, 
                                   computational_model, 
                                   args.time_window, 
                                   args.language, 
                                   args.average,
                                   args.corpus_portion,
                                   #limit
                                   )
                        )
if args.individuals:
    out_file = out_file.replace('gold', 'individuals')
    out_file = out_file.replace('_{}'.format(args.time_window), '')
if args.predicted:
    out_file = out_file.replace('gold', 'model')
if args.random:
    out_file = out_file.replace('gold', 'random')
with open(out_file, 'w') as o:
    o.write('subject\tentity\t{}_entity_vector\n'.format(computational_model))
    for k, v in tqdm(collector.items()):
        #tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        #full_data = [vec[1] for vec in entity_vectors[k]]
        #tsne_results = tsne.fit_transform(full_data)
        #fig, ax = pyplot.subplots(constrained_layout=True)
        #ax.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color='darkgray',zorder=1,)
        for sub, vec in v.items():
            ### closest neighbour
            #simz = {data[0] : scipy.stats.pearsonr(vec, data[1])[0] for data in entity_vectors[k]}
            #sorted_simz = sorted(simz.items(), key=lambda item : item[1], reverse=True)
            #best_fit = sorted_simz[0]
            #final_collector[k].extend([v[0] for v in entity_vectors[k] if v[0] in data])
            #final_collector[k].append([v[0] for v in entity_vectors[k] if v[0] in data])
            try:
                assert (len(vec),)  == out_shape
            except AssertionError:
                continue
            o.write('{}\t{}\t'.format(sub, k))
            for dim in vec:
                o.write('{}\t'.format(dim))
            o.write('\n')

            ### plotting vectors
            #color_data = [vec_i for vec_i, vec in enumerate(entity_vectors[k]) if vec[0] in data]

        #    ax.scatter(x=numpy.average(tsne_results[color_data][:, 0]), y=numpy.average(tsne_results[color_data][:, 1]), zorder=3)
        #ax.set_title(k)
        #pyplot.savefig(os.path.join('prova', '{}.jpg'.format(k)))
        #pyplot.clf()
        #pyplot.close()

'''
### finding the datapoints which are the closest 
with open(os.path.join(out_folder, 'exp_{}_{}_dirty_{}_vectors.tsv'.format(args.experiment_id, computational_model, args.language)), 'w') as o:
    o.write('subject\tentity\t{}_entity_vector\n'.format(computational_model))
    for k, v in tqdm(final_collector.items()):
        for sub in range(1, 34):
            #vecs = [numpy.average([vec[1] for vec in entity_vectors[k] if vec[0] in c_v], axis=0) for c_v in v]
            ### finding closest vectors to the average
            #for vec in vecs:
            vec = [vec[1] for vec in entity_vectors[k] if vec[0] in v]
            vec = numpy.average(vec, axis=0)
            try:
                assert vec.shape == out_shape
            except AssertionError:
                continue
            ##vec = numpy.average(random.sample(v, k=min(10, len(v))), axis=0)
            #vec = [vec[1] for vec in entity_vectors[k] if vec[0] in v]
#            vec = numpy.average(vec, axis=0)
#            assert vec.shape == out_shape
            o.write('{}\t{}\t'.format(sub, k))
            for dim in vec:
                o.write('{}\t'.format(dim))
            o.write('\n')
'''
