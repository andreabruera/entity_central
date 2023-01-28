import re

def return_entity_file(entity):
    entity = '{}{}'.format(entity[0].capitalize(), entity[1:])
    wiki_file = '{}.txt'.format(re.sub('[^a-zA-Z0-9]', '_', entity))
    return wiki_file
