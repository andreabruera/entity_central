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
from skbold.preproc import ConfoundRegressor
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from utils import load_comp_model_name, load_vec_files, read_args, read_brain_data, read_full_wiki_vectors

args = read_args(contextualized_selection=True)

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder = load_vec_files(args, computational_model)

entity_vectors, all_sentences = read_full_wiki_vectors(vecs_file, out_shape)

brain_data = read_brain_data(args)

if args.debugging:
    n_subjects = 2
else:
    n_subjects = len(brain_data.keys())

### preparing the folder

all_idxs = dict()

### per-subject feature selection
for s, sub_data in tqdm(brain_data.items()):

    current_entity_vectors = {k : v for k, v in entity_vectors.items() if k in sub_data.keys()}

    for k_one, k_two in zip(sorted(sub_data.keys()), sorted(current_entity_vectors.keys())):
        assert k_one == k_two

    sub_idxs = dict()
    ### Now actually doing the leave-one-out-evaluation
    for left_out in tqdm(sub_data.keys()):
        ### train
        train_data = [(data_point[1], sub_data[k], len(k), data_point[0]) for k, v in current_entity_vectors.items() for data_point in v if k!=left_out]
        train_input = [d[0] for d in train_data]
        train_target = [d[1] for d in train_data]
        train_lengths = [d[2] for d in train_data]
        #cfr = ConfoundRegressor(
        #                        confound=numpy.array(train_lengths), 
        #                        X=numpy.array(train_target),
        #                        cross_validate=True,
        #                        )
        #cfr.fit(numpy.array(train_target))
        #train_target = cfr.transform(numpy.array(train_target))
        ### test
        test_data = [(data_point[1], sub_data[k], len(k), data_point[0]) for k, v in current_entity_vectors.items() for data_point in v if k==left_out]
        test_input = [d[0] for d in test_data]
        test_target = [d[1] for d in test_data]
        test_identifiers = [d[3] for d in test_data]
        model = sklearn.linear_model.Ridge()
        model.fit(train_input, train_target)
        predictions = model.predict(test_input)
        assert len(predictions) == len(test_target)
        corrs = list()
        for pred, real in zip(predictions, test_target):
            corrs.append(scipy.stats.pearsonr(pred, real)[0])
        #idxs = sorted(enumerate(corrs), key=lambda item : item[1], reverse=True)
        idxs_data = [[test_id, test_in, val] for test_id, test_in, val in zip(test_identifiers, test_input, corrs)]
        sub_idxs[left_out] = idxs_data

    out_idxs = dict()
    for left_out in tqdm(sub_idxs.keys()):
        train_input = list()
        train_target = list()
        test_input = list()
        test_identifiers = list()
        for k, v in sub_idxs.items():
            if left_out == k:
                for item in v:
                    #test_input.append(numpy.hstack([item[1], sub_data[k]]))
                    test_input.append(item[1])
                    test_identifiers.append(item[0])
            else:
                for item in v:
                    #train_input.append(numpy.hstack([item[1], sub_data[k]]))
                    train_input.append(item[1])
                    train_target.append(item[2])
        model = sklearn.linear_model.Ridge()
        model.fit(train_input, train_target)
        predictions = model.predict(test_input)
        idxs = [k[0] for k in sorted(enumerate(predictions), key=lambda item : item[1], reverse=True)]
        out_idxs[left_out] = [test_identifiers[idx] for idx in idxs]

    with open(os.path.join(rankings_folder, 'sub-{:02}_model.ranking'.format(s)), 'w') as o:
        for stim, idxs in out_idxs.items():
            o.write('{}\t'.format(stim))
            for idx in idxs:
                o.write('{}\t'.format(idx))
            o.write('\n')
