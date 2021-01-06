import numpy
import os

def vectors(self, words):
    word_vectors = {w : extract_word_vectors(w) for w in words if re.sub('[0-9]', '', str(w)) != ''}
    return word_vectors

def prepare_file_name(entity):

    entity_one = re.sub(' ', '_', entity)
    entity_file_name = '{}.vec'.format(entity_one)
    return entity_file_name

def extract_word_vectors(entity):
    path = '/import/cogsci/andrea/dataset/word_vectors/bert_january_2020/bert_full_sentence_prova'
    entity_file_name = prepare_file_name(entity)

    try:
        with open(os.path.join(path, entity_file_name)) as entity_file:
            entity_lines = [l for l in entity_file.readlines()]
        stop = False
    except FileNotFoundError:
        mapping_dict = {'Object.vec' : 'Physical_object.vec'}
        try:
            with open(os.path.join(path, mapping_dict[entity_file_name])) as entity_file:
                entity_lines = [l for l in entity_file.readlines()]
            stop = False
        except KeyError:
            stop = True
        
    if not stop:

        lines = [l.split('\t')[0] for l in entity_lines]
        entity_vectors = [numpy.array(l.strip().split('\t')[1:], dtype=numpy.single) for l in entity_lines]
        if len(entity_vectors) > 24:
            entity_vectors = entity_vectors[:24]
    else:
        entity_vectors = list()

    return entity_vectors
