from scipy.stats.morestats import wilcoxon

import pickle
import numpy
import itertools
import re
import os
import scipy
import argparse
import math

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
from utils.brain_images import find_stable_voxels
from utils.general_utilities import load_word_vectors, print_wilcoxon
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

class fake_args():
    def __init__(self, voxels):
        self.amount_stable_voxels = int(voxels)

def correlation(one, two):
    corr = numpy.corrcoef(one, two)[0][1]
    if corr == numpy.nan:
        return 0.0
    else:
        return corr

def cos(v_one, v_two):
    sim = cosine_similarity(v_one.reshape(1, -1), v_two.reshape(1, -1))
    return sim

parser = argparse.ArgumentParser()
parser.add_argument('--uk', default=False, action='store_true', help='Tells whether to carry put the categorization analysis or not')
args = parser.parse_args()

#sizes = ['500', '1000', '5000', '50000', '68049']
sizes = ['500']
model = Ridge(alpha=1.0)

'''
ernie = load_word_vectors('ernie')
transe = (open(os.path.join(folder, 'transe_individuals.vec'), 'rb'))
w2v = pickle.load(open(os.path.join(folder, 'w2v_individuals.vec'), 'rb'))
bert = pickle.load(open(os.path.join(folder, 'bert_individuals.vec'), 'rb'))
wiki2vec = pickle.load(open(os.path.join(folder, 'wiki2vec_individuals.vec'), 'rb'))
transe_inverse = {k : v[0] for k, v in transe.items()}
transe_corrected = {v[0] : v[-1] for k, v in transe.items()}
wiki2vec_corrected = {transe_inverse[k] : v[-1] for k, v in wiki2vec.items()}

ernie_vectors = {k : ernie[k] for k in transe_corrected.keys()}
transe_vectors = {k : v for k, v in transe_corrected.items()}
w2v_vectors = {k : w2v[k] for k in transe_corrected.keys()}
bert_vectors = {k : bert[k] for k in transe_corrected.keys()}
wiki2vec_vectors = {k : v for k, v in wiki2vec_corrected.items()}
models = {'ernie' : ernie_vectors, 'transe' : transe_vectors, 'w2v' : w2v_vectors, 'bert' : bert_vectors, 'wiki2vec' : wiki2vec_vectors}
models_names = [k for k in models.keys()]
'''
#models_names = ['transe', 'word2vec_two', 'bert_two', 'wiki2vec', 'ernie_two']
models_names = ['transe', 'word2vec', 'bert', 'wiki2vec', 'ernie']
#models = {model : load_word_vectors(model)[0] for model in models_names}

men_models = ['word2vec', 'bert', 'wiki2vec']
#men_vectors = {model : load_word_vectors(model, men=True)[0] for model in men_models if model != 'transe' and 'ernie' not in model}
men_vectors = {model : load_word_vectors(model, categories=True)[0] for model in men_models if model != 'transe' and 'ernie' not in model}

men_comb = [k for k in itertools.combinations(men_models, 2)]

models_comb = [k for k in itertools.combinations(models_names, 2)]
subjects = [k for k in range(1, 17)]

# Entity vectors

evaluation = defaultdict(float)
'''
for c in tqdm(models_comb):
    final_results = []
    m_one = [v for k, v in models[c[0]].items()]
    keys = [k for k in models[c[0]].keys()]
    m_two = [models[c[1]][k] for k in keys]
    for k in range(len(m_one)):
        train_data = [[vec, m_two[i]] for i, vec in enumerate(m_one) if i != k]
        test_data = m_one[k]
        model.fit([item[0] for item in train_data], [item[1] for item in train_data])
        predictions = model.predict(test_data.reshape(1, -1))[0]
        results = defaultdict(float)
        for n, t in zip(keys, m_two):
            results[n] = cos(t, predictions)
        results_ordered = [(person_key, results[person_key]) for person_key in sorted(results, key = results.get, reverse = True)]
        rank = [i+1 for i, v in enumerate(results_ordered) if v[0] == keys[k]][0]
        accuracy = 1 - ((rank -1) / (len(m_two) - 1))
        final_results.append(accuracy)
    evaluation[c] = [numpy.median(final_results), numpy.average(final_results)]

    # RSA

    comb = [k for k in itertools.combinations([k for k in range(len(keys))], 2)]
    one = [models[c[0]][k] for k in keys]
    two = [models[c[1]][k] for k in keys]
    pairwise_one = [spearmanr(one[i[0]], one[i[1]])[0] for i in comb]
    pairwise_two = [spearmanr(two[i[0]], two[i[1]])[0] for i in comb]
    evaluation[c].append(spearmanr(pairwise_one, pairwise_two)[0])

with open('temp/vectors_decoding.txt', 'w') as o:
    for k, v in evaluation.items():
        o.write('Models: {}, {}\nMedian and average scores: {}, {}\nRSA correlation: {}\n\n'.format(k[0], k[1], v[0], v[1], v[2]))
'''
# Common nouns

evaluation = defaultdict(float)

for c in tqdm(men_comb):

    print(c)

    final_results = []
    m_one = [v for k, v in men_vectors[c[0]].items()]
    keys = [k for k in men_vectors[c[0]].keys()]
    m_two = [v for k, v in men_vectors[c[1]].items()]
    #m_two = [men_vectors[c[1]][k] for k in keys]
    assert len(m_one) == len(m_two)

    if args.uk:
        indices = defaultdict(list)
        indices['proper_names'] = [i for i in range(0, len(keys), 2)]
        indices['common_nouns'] = [i for i in range(1, len(keys), 2)]
        proper_names = [keys[i] for i in indices['proper_names']]
        common_nouns = [keys[i] for i in indices['common_nouns']]

        # Ridge regression for the UK setup only

        wilcoxon_results = defaultdict(list)

        for category, cat_indices in indices.items():

            for k in range(len(cat_indices)):
                relevant_one = [m_one[i] for i in cat_indices]
                relevant_two = [m_two[i] for i in cat_indices]
                train_data = [[vec, relevant_two[i]] for i, vec in enumerate(relevant_one) if i != k]
                test_data = relevant_one[k]
                model.fit([item[0] for item in train_data], [item[1] for item in train_data])
                predictions = model.predict(test_data.reshape(1, -1))[0]
                results = defaultdict(float)
                for n, t in zip(keys, relevant_two):
                    results[n] = cos(t, predictions)
                results_ordered = [(person_key, results[person_key]) for person_key in sorted(results, key = results.get, reverse = True)]
                rank = [i+1 for i, v in enumerate(results_ordered) if v[0] == keys[k]][0]
                accuracy = 1 - ((rank -1) / (len(relevant_two) - 1))
                final_results.append(accuracy)
                wilcoxon_results[category].append(accuracy)
            print([category, numpy.median(final_results), numpy.average(final_results)])

            # RSA

            comb = [k for k in itertools.combinations([k for k in range(len(cat_indices))], 2)]
            one = [m_one[i] for i in cat_indices]
            two = [m_two[i] for i in cat_indices]
            pairwise_one = [spearmanr(one[i[0]], one[i[1]])[0] for i in comb]
            pairwise_two = [spearmanr(two[i[0]], two[i[1]])[0] for i in comb]
            print(spearmanr(pairwise_one, pairwise_two)[0])

        # Checking statistical significance with the Wilcoxon test
        max_length = min([len(l) for k, l in wilcoxon_results.items()])
        z_value, p_value = wilcoxon(wilcoxon_results['proper_names'][:max_length], wilcoxon_results['common_nouns'][:max_length])
        effect_size = abs(z_value / math.sqrt(max_length))
        print('P value: {} - effect size: {}'.format(p_value, effect_size))
        '''
        with open('temp/uk_vectors_decoding.txt', 'w') as o:
            for k, v in evaluation.items():
                for cat in v:
                    o.write('Models: {}, {}\n{}: Median and average scores: {}, {}\nRSA correlation: {}\n\n'.format(k[0], k[1], v[0], v[1], v[2]))
        '''

    else: 
        # Ridge regression

        for k in range(len(m_one)):
            train_data = [[vec, m_two[i]] for i, vec in enumerate(m_one) if i != k]
            test_data = m_one[k]
            model.fit([item[0] for item in train_data], [item[1] for item in train_data])
            predictions = model.predict(test_data.reshape(1, -1))[0]
            results = defaultdict(float)
            for n, t in zip(keys, m_two):
                results[n] = cos(t, predictions)
            results_ordered = [(person_key, results[person_key]) for person_key in sorted(results, key = results.get, reverse = True)]
            rank = [i+1 for i, v in enumerate(results_ordered) if v[0] == keys[k]][0]
            accuracy = 1 - ((rank -1) / (len(m_two) - 1))
            final_results.append(accuracy)
        evaluation[c] = [numpy.median(final_results), numpy.average(final_results)]

        # RSA

        comb = [k for k in itertools.combinations([k for k in range(len(keys))], 2)]
        one = [men_vectors[c[0]][k] for k in keys]
        two = [men_vectors[c[1]][k] for k in keys]
        pairwise_one = [spearmanr(one[i[0]], one[i[1]])[0] for i in comb]
        pairwise_two = [spearmanr(two[i[0]], two[i[1]])[0] for i in comb]
        evaluation[c].append(spearmanr(pairwise_one, pairwise_two)[0])

    with open('temp/men_vectors_decoding.txt', 'w') as o:
        for k, v in evaluation.items():
            o.write('Models: {}, {}\nMedian and average scores: {}, {}\nRSA correlation: {}\n\n'.format(k[0], k[1], v[0], v[1], v[2]))

rsa_final = defaultdict(list)

sub_comb = [k for k in itertools.combinations(subjects, 2)]

'''
for c in sub_comb:
    one_wrong = pickle.load(open('pickles/fmri_sub_{:02}.pkl'.format(c[0]), 'rb'))
    one_reduced, mask = find_stable_voxels(fake_args(500), one_wrong)
    two_wrong = pickle.load(open('pickles/fmri_sub_{:02}.pkl'.format(c[1]), 'rb'))
    two_reduced, mask_two = find_stable_voxels(fake_args(500), two_wrong)
    keys = [k for k in one_wrong.keys() if k in two_wrong.keys()]
    two = {k : two_reduced[k] for k in keys}
    one = {k : one_reduced[k] for k in keys}
    comb = [k for k in itertools.combinations(keys, 2)]
    pairwise_one = [spearmanr(one[k[0]][0], one[k[1]][0])[0] for k in comb]
    pairwise_two = [spearmanr(two[k[0]][0], two[k[1]][0])[0] for k in comb]
    #pairwise_two = [spearmanr(v[0], v[1])[0] for k, v in two.items()]
    #print(pairwise_one)
    #print(c, numpy.corrcoef(pairwise_one, pairwise_two[0][1])
    #correlation = numpy.corrcoef(pairwise_one, pairwise_two)[0][1]
    correlation = spearmanr(pairwise_one, pairwise_two)[0]
    print(c, correlation)
'''
'''
for voxel_size in sizes:
    print(re.sub('_', '', voxel_size))
    for s in tqdm(subjects):
        pickled_sub = pickle.load(open('pickles/fmri_sub_{:02}.pkl'.format(s), 'rb'))
        #fmri = {transe_inverse[k] : v for k, v in pickled_sub.items()}
        fmri_reduced, mask = find_stable_voxels(fake_args(voxel_size), pickled_sub)
        #fmri_vecs = [v for k, v in fmri_selected.items()]
        fmri_vecs = [v[0] for k, v in fmri_reduced.items()]
        fmri_keys = [k for k in fmri_reduced.keys()]
        comb = [k for k in itertools.combinations([k for k in range(len(fmri_keys))], 2)]
        for k, m_full in models.items():
            m = [m_full[k] for k in fmri_keys]
            pairwise_comp = [numpy.corrcoef(m[i[0]], m[i[1]])[0][1] for i in comb]
            #pairwise_comp = [spearmanr(m[i[0]], m[i[1]])[0] for i in comb]
            #pairwise_fmri = [numpy.corrcoef(fmri_vecs[i[0]][0], fmri_vecs[i[1]][0])[0][1] for i in comb]
            pairwise_fmri = [numpy.corrcoef(fmri_vecs[i[0]], fmri_vecs[i[1]])[0][1] for i in comb]
            #pairwise_fmri = [spearmanr(fmri_vecs[i[0]][0], fmri_vecs[i[1]][0])[0] for i in comb]
            rsa_score = numpy.corrcoef(pairwise_comp, pairwise_fmri)[0][1]
            #rsa_score = spearmanr(pairwise_comp, pairwise_fmri)[0]
            #print(rsa_score)
            #rsa_final['{}_{}'.format(k, voxel_size)].append(rsa_score)
            rsa_final['{}'.format(k)].append(rsa_score)
        #print(numpy.nanmean(rsa_final['{}_{}'.format(k, voxel_size)]))
        print(numpy.nanmean(rsa_final['{}'.format(k)]))

os.makedirs('results', exist_ok=True)
with open(os.path.join('results', 'rsa_results_pre_decoding.txt'), 'w') as o:
    for k, scores in rsa_final.items():
        print('Average, median and std for {}: {}, {}, {}'.format(k, numpy.nanmean(scores), numpy.nanmedian(scores), numpy.nanstd(scores)))
        o.write('Average, median and std for {}: {}, {}, {}\n\n'.format(k, numpy.nanmean(scores), numpy.nanmedian(scores), numpy.nanstd(scores)))

'''
'''
#comb = [k for k in itertools.combinations([k for k in range(len(models['transe'].keys()))], 2)]
keys = [k for k, v in load_word_vectors('word2vec_two')[0].items() if 0.0 not in v]
comb = [k for k in itertools.combinations([k for k in range(len(keys))], 2)]

with open(os.path.join('results', 'rsa_results_nlp_models_spearman.txt'), 'w') as o:
    for c in models_comb:
        one = [models[c[0]][k] for k in keys]
        two = [models[c[1]][k] for k in keys]
        #two = [v for k, v in models[c[1]].items()]

        #pairwise_one = [numpy.corrcoef(one[i[0]], one[i[1]])[0][1] for i in comb]
        #pairwise_two = [numpy.corrcoef(two[i[0]], two[i[1]])[0][1] for i in comb]
        pairwise_one = [spearmanr(one[i[0]], one[i[1]])[0] for i in comb]
        pairwise_two = [spearmanr(two[i[0]], two[i[1]])[0] for i in comb]
        #print(c, numpy.corrcoef(pairwise_one, pairwise_two[0][1])
        #correlation = numpy.corrcoef(pairwise_one, pairwise_two)[0][1]
        correlation = spearmanr(pairwise_one, pairwise_two)[0]
        print(c, correlation)
        #o.write('{}\n{}\n\n'.format(c, numpy.corrcoef(pairwise_one, pairwise_two)[0][1]))
        o.write('{}\n{}\n\n'.format(c, correlation))
'''
