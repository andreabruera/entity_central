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
from sklearn.linear_model import Ridge, RidgeCV, RidgeClassifier
from tqdm import tqdm

from utils import load_comp_model_name, load_vec_files, read_args, read_brain_data, read_full_wiki_vectors
from exp_two_utils import read_exp_two_data

args = read_args(contextualized_selection=True)

model_name, computational_model, out_shape = load_comp_model_name(args)

vecs_file, rankings_folder = load_vec_files(args, computational_model)

entity_vectors, all_sentences = read_full_wiki_vectors(vecs_file, out_shape)

brain_data = read_brain_data(args)

if args.debugging:
    n_subjects = 2
else:
    n_subjects = len(brain_data.keys())

if args.individuals:
    ### per-subject feature selection
    #for s, sub_data in tqdm(brain_data.items()):
    #ents = {k for s, sub_data in brain_data.items() for k in sub_data.keys()}


    #entity_vectors = {k : v for k, v in entity_vectors.items() if k in ents}

    #sub_idxs = dict()
    ent_idxs = dict()
    ### Now actually doing the leave-one-out-evaluation
    if args.experiment_id == 'two':
        poss = ['{:02}_'.format(i) for i in range(1, 34)]
    else:
        poss = ['']
    for sub_marker in tqdm(poss):
        #for left_out in tqdm(labels_mapper.keys()):
        with tqdm() as counter:
            for left_out in entity_vectors.keys():
                if args.experiment_id == 'two':
                    possible_sub = re.findall('^\d\d_', left_out)
                    if len(possible_sub) == 1:
                        if possible_sub[0] != sub_marker:
                            continue
                        else:
                            left_out_marker = '{}'.format(left_out)
                    elif len(possible_sub) == 0:
                        left_out_marker = '{}{}'.format(sub_marker, left_out)
                    candidate_vectors = dict()
                    for k, v in entity_vectors.items():
                        possible_sub = re.findall('^\d\d_', k)
                        if len(possible_sub) == 1:
                            if possible_sub[0] == sub_marker:
                                candidate_vectors[k] = v
                            else:
                                continue
                        elif len(possible_sub) == 0:
                            candidate_vectors[k] = v
                    assert len(candidate_vectors) == 34
                        
                else:
                    #left_out_marker = '{}'.format(left_out)
                    candidate_vectors = entity_vectors.copy()
                    left_out_marker = '{}{}'.format(sub_marker, left_out)
                data_labels = [(k, vec) for k, v in candidate_vectors.items() for vec in v]
                #print(left_out_marker)
                if args.experiment_id == 'two':
                    ### reordering
                    reaction_times, coarse, fame = read_exp_two_data(sub_marker)
                    ### using reaction times
                    labels_mapper = reaction_times.copy()
                    ### order:
                    ### famous -> familiar
                    ### people - place - place - person
                    ordered_names = [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='person' and fa[1]=='familiar'] + \
                            [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='person' and fa[1]=='famous'] + \
                            [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='place' and fa[1]=='familiar'] + \
                            [co[0] for co, fa in zip(coarse.items(), fame.items()) if co[1]=='place' and fa[1]=='famous']
                    labels_mapper = [(k, k_i) for k_i, k in enumerate(ordered_names)]
                    labels = [k[1] for k in labels_mapper]
                    names = [k[0] for k in labels_mapper]
                    norm_labels = [int(2*((x-min(labels))/(max(labels)-min(labels)))-1) for x in labels]
                    assert min(norm_labels) == -1
                    assert max(norm_labels) == 1
                    labels_mapper = {n : l for n, l in zip(names, labels)}

                    #print(labels_mapper)
                    assert len(labels_mapper.keys()) == 32
                    candidate_vectors = {k : v for k, v in candidate_vectors.items() if k in labels_mapper.keys()}
                    assert len(candidate_vectors.keys()) == 32
                    #labels_mapper = {k : k_i for k_i, k in enumerate(candidate_vectors.keys())}
                else:
                    labels_mapper = {k : k_i for k_i, k in enumerate(candidate_vectors.keys())}
                if left_out not in candidate_vectors.keys():
                    print(left_out)
                    continue
                ### train
                train_data = [(data_point[1], labels_mapper[k], len(k), data_point[0]) for k, v in candidate_vectors.items() for data_point in v if k!=left_out]
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
                test_data = [(data_point[1], labels_mapper[k], len(k), data_point[0]) for k, v in candidate_vectors.items() for data_point in v if k==left_out]
                test_input = [d[0] for d in test_data]
                test_target = [d[1] for d in test_data]
                test_identifiers = [d[3] for d in test_data]
                model = sklearn.linear_model.RidgeCV()
                #model = sklearn.linear_model.RidgeClassifier()
                model.fit(train_input, train_target)
                predictions = model.predict(test_input)
                #predictions = model.decision_function(test_input)
                #predictions = scipy.stats.zscore(predictions, axis=0)
                #corrs = [pred[test_target[0]] for pred in predictions]
                assert len(predictions) == len(test_target)
                corrs = list()
                for pred, real in zip(predictions, test_target):
                    #corrs.append(scipy.stats.pearsonr(pred, real)[0])
                    corrs.append(abs(real-pred))
                #idxs = ['{},{}'.format(test_identifiers[k[0]], k[1]) for k in sorted(enumerate(corrs), key=lambda item : item[1])]
                idxs = ['{},{}'.format(
                                       test_identifiers[k[0]], k[1]) for k in sorted(enumerate(corrs), 
                                       key=lambda item : item[1],
                                       #reverse=True,
                                       )
                                       ]
                ent_idxs[left_out_marker] = idxs
                counter.update(1)

    print(rankings_folder)
    with open(os.path.join(rankings_folder, 'individuals.ranking'), 'w') as o:
        for stim, idxs in ent_idxs.items():
            o.write('{}\t'.format(stim))
            for idx in idxs:
                o.write('{}\t'.format(idx))
            o.write('\n')
    #for s, sub_data in brain_data.items():
        #for k_one, k_two in zip(sorted(sub_data.keys()), sorted(ent_idxs.keys())):
        #    assert k_one == k_two
        #with open(os.path.join(rankings_folder, 'sub-{:02}_individuals.ranking'.format(s)), 'w') as o:
            #for stim, idxs in ent_idxs.items():
            #    if stim in sub_data.keys():
            #        o.write('{}\t'.format(stim))
            #        for idx in idxs:
            #            o.write('{}\t'.format(idx))
            #        o.write('\n')
            #    else:
            #        print(stim)

elif args.predicted:
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

        with open(os.path.join(rankings_folder, 'sub-{:02}_predicted.ranking'.format(s)), 'w') as o:
            for stim, idxs in out_idxs.items():
                o.write('{}\t'.format(stim))
                for idx in idxs:
                    o.write('{}\t'.format(idx))
                o.write('\n')

else:
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
            idxs = [test_identifiers[k[0]] for k in sorted(enumerate(corrs), key=lambda item : item[1], reverse=True)]
            sub_idxs[left_out] = idxs
        with open(os.path.join(rankings_folder, 'sub-{:02}_gold.ranking'.format(s)), 'w') as o:
            for stim, idxs in sub_idxs.items():
                o.write('{}\t'.format(stim))
                for idx in idxs:
                    o.write('{}\t'.format(idx))
                o.write('\n')
