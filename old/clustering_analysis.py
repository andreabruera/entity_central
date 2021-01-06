import collections
import os
import itertools
import sklearn

import numpy

from utils.tsne_plots import tsne_plot_words
from utils.general_utilities import load_word_vectors
from sklearn.cluster import KMeans
from tqdm import tqdm


def purity(binary_labels_list, number_samples):
    numerator = sum(binary_labels_list) if sum(binary_labels_list) > (number_of_vectors / 2) else number_of_vectors - sum(binary_labels_list)
    
    purity_value = round(numerator / number_of_vectors, 3)
    return purity_value

data_categories = ['people', 'places']
results_all = collections.defaultdict(dict)
results_binary = collections.defaultdict(dict)

for d in data_categories:

    category_results_all = collections.defaultdict(list)
    category_results_binary = collections.defaultdict(list)

    base_folder = os.path.join(os.getcwd(), 'categorized_word_vectors', d)
    types = os.listdir(base_folder)
    for t in tqdm(types):
        vecs = collections.defaultdict(list)
        current_folder = os.path.join(base_folder, t)

        # Clustering among all individuals

        for root, directories, files in os.walk(current_folder): 
            #combs = [i for i in itertools.combinations(files, 2)]
            #for c in combs:
                #with open(os.path.join(root, c[0])) as input_file:
                    #lines_one = [l.strip().split('\t')[1:] for l in input_file.readlines()]
                    #len_one = len(lines_one)
                #with open(os.path.join(root, c[1])) as input_file:
                    #lines_two = [l.strip().split('\t')[1:] for l in input_file.readlines()]
                    #len_two = len(lines_two)
            for f in files:
                with open(os.path.join(root, f)) as input_file:
                    lines = [l.strip().split('\t')[1:] for l in input_file.readlines()]
                    vecs[f] = lines

        number_of_vectors = min([len(v) for k, v in vecs.items()])
        vecs_final = {k : numpy.asarray(lines[:number_of_vectors], dtype=numpy.single) for k, lines in vecs.items()}
        #vecs_one = numpy.asarray(lines_one[:number_of_vectors], dtype=numpy.single)
        #vecs_two = numpy.asarray(lines_two[:number_of_vectors], dtype=numpy.single)
            
        ### K-Means clustering

        n_clusters = len(vecs_final)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        try:
            kmeans.fit([v_two for k, v in vecs_final.items() for v_two in v])
            good_labels = [k for k in range(n_clusters) for k_two in range(number_of_vectors)]
            category_results_all[t].append(sklearn.metrics.v_measure_score(good_labels, kmeans.labels_))
            #kmeans.fit(vecs_one + vecs_two)
            #kmeans.fit([v for k, v in vecs_final.items()])
            #import pdb; pdb.set_trace()
            '''
            class_one = kmeans.labels_[:number_of_vectors]
            class_two = kmeans.labels_[number_of_vectors:]
            purity_one = purity(class_one, number_of_vectors)
            purity_two = purity(class_two, number_of_vectors)
            category_results[t].append(purity_one)
            category_results[t].append(purity_two)
            '''
        except ValueError:
            #print(c)
            print(d)

        vecs = collections.defaultdict(list)

        # Clustering among two individuals

        for root, directories, files in os.walk(current_folder): 
            combs = [i for i in itertools.combinations(files, 2)]
            for c in combs:
                with open(os.path.join(root, c[0])) as input_file:
                    lines_one = [l.strip().split('\t')[1:] for l in input_file.readlines()]
                    len_one = len(lines_one)
                with open(os.path.join(root, c[1])) as input_file:
                    lines_two = [l.strip().split('\t')[1:] for l in input_file.readlines()]
                    len_two = len(lines_two)

                number_of_vectors = min([len_one, len_two])
                #vecs_final = {k : numpy.asarray(lines[:number_of_vectors], dtype=numpy.single) for k, lines in vecs.items()}
                vecs_one = lines_one[:number_of_vectors]
                vecs_two = lines_two[:number_of_vectors]
                    
                ### K-Means clustering

                n_clusters = 2

                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                try:
                    kmeans.fit(vecs_one + vecs_two)
                    #kmeans.fit([v_two for k, v in vecs_final.items() for v_two in v])
                    good_labels = [k for k in range(n_clusters) for k_two in range(number_of_vectors)]
                    category_results_binary[t].append(sklearn.metrics.v_measure_score(good_labels, kmeans.labels_))
                    #kmeans.fit([v for k, v in vecs_final.items()])
                    #import pdb; pdb.set_trace()
                    '''
                    class_one = kmeans.labels_[:number_of_vectors]
                    class_two = kmeans.labels_[number_of_vectors:]
                    purity_one = purity(class_one, number_of_vectors)
                    purity_two = purity(class_two, number_of_vectors)
                    category_results[t].append(purity_one)
                    category_results[t].append(purity_two)
                    '''
                except ValueError:
                    #print(c)
                    print(d)

    results_binary[d] = category_results_binary
    results_all[d] = category_results_all
                ### Plotting

                #title = 'TSNE plot comparing place and person vectors in {}'.format(m)
                #labels = ['People - pur {}'.format(purity_people), 'Places - pur {}'.format(purity_places)]
                #labels = ['', '']
                #filename = 'temp/tsne_{}_{}.png'.format(data_type, m)

                #tsne_plot_words(title, final_dict['people'], final_dict['places'], labels, filename)

with open('temp/clustering_results_binary.txt', 'w') as o:
    for k_one, v_one in results_binary.items():
        o.write('{}\n\n'.format(k_one))
        for k_two, v_two in v_one.items():
            o.write('{}\n'.format(k_two))
            o.write('Average:\t{}\nMedian\t:{}\nStandard deviation:\t{}\nAmount of evaluations:\t{}\n\n'.format(numpy.average(v_two), numpy.median(v_two), numpy.std(v_two), len(v_two)))

with open('temp/clustering_results_all.txt', 'w') as o:
    for k_one, v_one in results_all.items():
        o.write('{}\n\n'.format(k_one))
        for k_two, v_two in v_one.items():
            o.write('{}\n'.format(k_two))
            o.write('V-score: {}\n\n'.format(v_two))
