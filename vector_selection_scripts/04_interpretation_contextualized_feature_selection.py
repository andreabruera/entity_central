import argparse
import itertools
import joblib
import multiprocessing
import numpy
import os
import random
import re
import scipy
import sklearn

from nilearn import image, masking
from scipy import stats
from skbold.preproc import ConfoundRegressor
from sklearn.linear_model import Ridge, RidgeCV
from tqdm import tqdm

from utils import load_comp_model_name, load_vec_files, read_args, read_full_wiki_vectors

def predict_brain(arguments):

    train_input = arguments[0]
    train_target = arguments[1]
    t_in = arguments[2]
    t_target = arguments[3]
    idx = arguments[4]

    internal_corrs = [scipy.stats.pearsonr(t_in, tr)[0] for tr in train_input]
    ### Encoding targets (brain images)
    pred = numpy.nansum([t*internal_corrs[t_i] for t_i, t in enumerate(train_target)], axis=0)
    corr = scipy.stats.pearsonr(pred, t_target)[0]

    return((idx, corr))

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
    if args.debugging:
        current_entity_vectors = {k : v[:100] for k, v in current_entity_vectors.items()}
        current_entity_vectors = {k : current_entity_vectors[k] for k in list(current_entity_vectors.keys())[:2]}
        sub_data = {k : sub_data[k] for k in list(current_entity_vectors.keys())[:2]}

    for k_one, k_two in zip(sorted(sub_data.keys()), sorted(current_entity_vectors.keys())):
        assert k_one == k_two

    sub_idxs = dict()
    all_train_data = [(data_point[1], sub_data[k], len(k), k, data_point[0]) for k, v in current_entity_vectors.items() for data_point in v]
    ### Now actually doing the leave-one-out-evaluation
    for outer_test_item in tqdm(sub_data.keys()):
        outer_test_identifiers = [data_point[4] for data_point in all_train_data if data_point[3]==outer_test_item]
        scores = {k : list() for k in outer_test_identifiers}
        for inner_test_item in tqdm(sub_data.keys()):
            if inner_test_item == outer_test_item:
                continue
            ### train
            train_data = [data_point for data_point in all_train_data if data_point[3] not in [inner_test_item] and numpy.sum(data_point[0])>0.]
            train_input = [d[0] for d in train_data]
            train_target = [d[1] for d in train_data]
            train_lengths = [d[2] for d in train_data]

            outer_test_idxs = [d_i for d_i, data_point in enumerate(train_data) if data_point[3]==outer_test_item]

            ### rand

            ### correcting for word length
            #if args.corrected:
            '''
            cfr = ConfoundRegressor(
                                    confound=numpy.array(train_lengths), 
                                    X=numpy.array(train_target),
                                    cross_validate=True,
                                    )
            cfr.fit(numpy.array(train_target))
            train_target = cfr.transform(numpy.array(train_target))
            '''
            ### test
            test_data = [data_point for data_point in all_train_data if data_point[3] == inner_test_item and numpy.sum(data_point[0])>0.]
            #test_data = [(data_point, sub_data[k]) for k, v in current_entity_vectors.items() for data_point in v if k==inner_test_item]
            test_input = [d[0] for d in test_data]
            test_target = [d[1] for d in test_data]
            ### training the full model
            model = sklearn.linear_model.Ridge()
            model.fit(train_input, train_target)
            predictions = model.predict(test_input)
            assert len(predictions) == len(test_target)
            full_corrs = list()
            for pred, real in zip(predictions, test_target):
                full_corrs.append(scipy.stats.pearsonr(pred, real)[0])

            ### now removing items...
            for i in tqdm(range(100)):
                ### blanking out 5% of the input
                leave_out_idxs = random.sample(outer_test_idxs, k=int(len(outer_test_idxs)/20))
                current_identifiers = [outer_test_identifiers[outer_test_idxs.index(idx)] for idx in leave_out_idxs]
                current_train_input = [d[0] for d_i, d in enumerate(train_data) if d_i not in leave_out_idxs]
                current_train_target = [d[1] for d_i, d in enumerate(train_data) if d_i not in leave_out_idxs]
                current_train_lengths = [d[2] for d_i, d in enumerate(train_data) if d_i not in leave_out_idxs]
                assert len(current_train_input) < len(train_input)
                ### training the full model
                model = sklearn.linear_model.Ridge()
                model.fit(current_train_input, current_train_target)
                predictions = model.predict(test_input)
                assert len(predictions) == len(test_target)
                current_corrs = list()
                for pred, real in zip(predictions, test_target):
                    current_corrs.append(scipy.stats.pearsonr(pred, real)[0])
                current_diff = numpy.average(numpy.array(full_corrs) - numpy.array(current_corrs))
                for idx in current_identifiers:
                    scores[idx].append(current_diff)

        scores = {k : numpy.average(v) for k, v in scores.items()}
        idxs = [k[0] for k in sorted(scores.items(), key=lambda item : item[1])]
        sub_idxs[outer_test_item] = idxs

    with open(os.path.join(rankings_folder, 'sub-{:02}.ranking'.format(s)), 'w') as o:
        for stim, idxs in sub_idxs.items():
            o.write('{}\t'.format(stim))
            for idx in idxs:
                o.write('{}\t'.format(idx))
            o.write('\n')
