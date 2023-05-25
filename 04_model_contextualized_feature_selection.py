import argparse
import itertools
import joblib
import random
import numpy
import os
import re
import scipy
import sklearn

from nilearn import image, masking
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from utils import load_comp_model_name, load_vec_files, read_args, read_full_wiki_vectors

args = read_args()

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder = load_vec_files(args, computational_model)

entity_vectors, all_sentences = read_full_wiki_vectors(vecs_file, out_shape)

### now ranking sentences for each subject!
### Loading brain data
brain_data_file = os.path.join('brain_data', args.experiment_id, 'rough_and_ready_brain_data_exp_{}.eeg'.format(args.experiment_id))
assert os.path.exists(brain_data_file)
with open(brain_data_file) as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]
subjects = set([int(l[0]) for l in lines])
brain_data = {sub : {l[1] : numpy.array(l[2:], dtype=numpy.float64) for l in lines if int(l[0])==sub} for sub in subjects}

if args.debugging:
    n_subjects = 2
else:
    n_subjects = len(subjects)

### preparing the folder

all_idxs = dict()

### per-subject feature selection
for s, sub_data in tqdm(brain_data.items()):

    current_entity_vectors = {k : v for k, v in entity_vectors.items() if k in sub_data.keys()}

    for k_one, k_two in zip(sorted(sub_data.keys()), sorted(current_entity_vectors.keys())):
        assert k_one == k_two

    sub_idxs = dict()
    ### Now actually doing the leave-one-out-evaluation
    for outer_loop_item in tqdm(sub_data.keys()):
        outer_train_input = list()
        outer_train_target = list()
        outer_test_data = [(data_point[1], sub_data[k], len(k), data_point[0]) for k, v in current_entity_vectors.items() for data_point in v if k==outer_loop_item]
        outer_test_input = [d[0] for d in outer_test_data]
        outer_test_target = [d[1] for d in outer_test_data]
        outer_test_identifiers = [d[3] for d in outer_test_data]

        for inner_loop_item in tqdm(sub_data.keys()):
            if inner_loop_item == outer_loop_item:
                continue
            ### train
            train_data = [(data_point[1], sub_data[k], len(k)) for k, v in current_entity_vectors.items() for data_point in v if k not in [inner_loop_item, outer_loop_item]]
            train_input = [d[0] for d in train_data]
            train_target = [d[1] for d in train_data]
            ### test
            test_data = [(data_point[1], sub_data[k], len(k)) for k, v in current_entity_vectors.items() for data_point in v if k==inner_loop_item]
            test_input = [d[0] for d in test_data]
            test_target = [d[1] for d in test_data]
            model = sklearn.linear_model.Ridge()
            model.fit(train_input, train_target)
            predictions = model.predict(test_input)
            assert len(predictions) == len(test_target)
            corrs = list()
            for pred, real in zip(predictions, test_target):
                corrs.append(scipy.stats.pearsonr(pred, real)[0])
            #idxs = [k[0] for k in sorted(enumerate(corrs), key=lambda item : item[1], reverse=True)]
            #sub_idxs[left_out] = idxs
            outer_train_input.extend(test_input)
            outer_train_target.extend(corrs)

        ### now, for each stimulus, we train a ridge regression model and predict scores for the unseen data
        ### predicting scores for unseen outer loop items
        model = sklearn.linear_model.Ridge()
        model.fit(outer_train_input, outer_train_target)
        corrs = model.predict(outer_test_input)
        idxs = [k[0] for k in sorted(enumerate(corrs), key=lambda item : item[1], reverse=True)]
        sub_idxs[outer_loop_item] = [outer_test_identifiers[idx] for idx in idxs]

    with open(os.path.join(rankings_folder, 'sub-{:02}.ranking'.format(s)), 'w') as o:
        for stim, idxs in sub_idxs.items():
            o.write('{}\t'.format(stim))
            for idx in idxs:
                o.write('{}\t'.format(idx))
            o.write('\n')
