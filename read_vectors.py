import numpy
import os
import re

class EntityVectors:

    def __init__(self, entities_dict, model_name, extraction_mode, max_number=24):

        self.model_name = model_name
        self.extraction_mode = extraction_mode
        self.max_number = max_number
        print('Now loading word vectors...')
        all_vectors = self.all_vectors(entities_dict.keys())
        self.vectors = {k : v for k, v in all_vectors.items() if len(v) > 0}
        self.to_be_deleted = [k for k, v in all_vectors.items() if len(v) == 0]

    def all_vectors(self, entities_list):
        word_vectors = {w : self.read_word_vector(w) for w in entities_list if re.sub('[0-9]', '', str(w)) != ''}
        return word_vectors

    def prepare_file_name(self, entity):

        entity = re.sub(' ', '_', entity)
        entity = '{}.vec'.format(entity)
        return entity

    def read_word_vector(self, entity):

        path = '/import/cogsci/andrea/dataset/word_vectors/{}/{}'.format(self.model_name, self.extraction_mode)
        entity_file_name = self.prepare_file_name(entity)

        try:
            with open(os.path.join(path, entity_file_name)) as entity_file:
                entity_lines = [l.strip().split('\t')[1:] for l in entity_file.readlines()]
            if 'facet' not in self.extraction_mode:
                entity_vectors = [numpy.array([value for value in l if not re.search('[a-z]+.+[a-z]+', value)], dtype=numpy.single) for l in entity_lines]
                for v in entity_vectors:
                    assert len(v) == 768
            else:
                entity_vectors = entity_lines.copy()
                for v in entity_vectors:
                    assert len(v) == 40
            ### reducing the amount of vectors to max_number
            entity_vectors = entity_vectors[:self.max_number]

        except FileNotFoundError:
            entity_vectors = list()

        return entity_vectors
