import numpy
import collections

from extract_word_lists import Entities
from read_vectors import EntityVectors

from scipy import stats
from sklearn import feature_extraction

extraction_mode = 'unmasked'
max_number = 24 

entities = Entities('wikisrs')
all_vectors = EntityVectors(entities_dict=entities.word_categories, model_name='bert', extraction_mode=extraction_mode, max_number=max_number).vectors


if extraction_mode == 'facets':

    vectorizer = feature_extraction.DictVectorizer()
    tfidf = feature_extraction.text.TfidfTransformer()

    print('Transforming the vectors')
    trans_vectors = numpy.array([vec for k, v in all_vectors.items() for vec in v[:max_number]])
    unraveled_entities = [k for k, v in all_vectors.items() for vec in v[:max_number]]
    del all_vectors

    vectorized = vectorizer.fit_transform(trans_vectors)
    print('Running tf-idf on the vectors')
    tf_idf_vecs = tfidf.fit_transform(vectorized).todense()

    print('Now returning to the dictionary form')
    all_vectors = collections.defaultdict(list)
    for i, ent in enumerate(unraveled_entities):
        vec = tf_idf_vecs[i, :].getA().flatten()
        all_vectors[ent].append(vec)

all_vectors = {k : numpy.average(v, axis=0) for k, v in all_vectors.items()}

### Similarity score
gold_sims = list()
predicted_sims = list()

with open('resources/WikiSRS/WikiSRS_similarity.csv') as input_file:
    lines = [l.strip().split('\t')[2:5] for l in input_file.readlines()][1:]
for l in lines:
    if l[0] in all_vectors.keys() and l[1] in all_vectors.keys():
         vec_one = all_vectors[l[0]]
         vec_two = all_vectors[l[1]]
         rho = stats.pearsonr(vec_one, vec_two)[0]
         predicted_sims.append(rho)
         gold_sims.append(float(l[2]))

corr = stats.pearsonr(gold_sims, predicted_sims)
print('Pearson correlation: {}'.format(corr[0]))
corr = stats.spearmanr(gold_sims, predicted_sims)
print('Spearman correlation: {}'.format(corr[0]))

### Relatedness score
gold_sims = list()
predicted_sims = list()
with open('resources/WikiSRS/WikiSRS_relatedness.csv') as input_file:
    lines = [l.strip().split('\t')[2:5] for l in input_file.readlines()][1:]

for l in lines:
    if l[0] in all_vectors.keys() and l[1] in all_vectors.keys():
         vec_one = all_vectors[l[0]]
         vec_two = all_vectors[l[1]]
         rho = stats.pearsonr(vec_one, vec_two)[0]
         predicted_sims.append(rho)
         gold_sims.append(float(l[2]))

corr = stats.pearsonr(gold_sims, predicted_sims)
print('Pearson correlation: {}'.format(corr[0]))
corr = stats.spearmanr(gold_sims, predicted_sims)
print('Spearman correlation: {}'.format(corr[0]))
