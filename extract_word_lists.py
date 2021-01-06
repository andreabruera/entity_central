import os
import re
import collections
import numpy

class Entities:

    def __init__(self, required_words):

        self.base_folder = '/import/cogsci/andrea/github/names_count'

        if required_words == 'wakeman_henson':
            self.words = self.wakeman_henson()
        elif required_words == 'full_wiki':
            self.words = self.full_wiki()
        elif required_words == 'men':
            self.words = self.men()
        elif required_words == 'eeg_stanford':
            self.words = self.eeg_stanford()
        elif required_words == 'mitchell':
            self.words = self.mitchell()
        elif required_words == 'stopwords':
            self.words = self.stopwords()

       
    def wakeman_henson(self):
        with open('resources/wakeman_henson_stimuli.txt') as input_file:
            lines = [l.strip().split('\t') for l in input_file.readlines()]
        names = [l[1] for l in lines if len(l) > 2]
        names_and_cats = {l[1] : (l[2], 'Person') for l in lines if len(l) > 2}
        return names_and_cats

    def eeg_stanford(self):
        #with open('/import/cogsci/andrea/github/fame/data/resources/eeg_data_ids.txt') as ids_txt:
        with open('resources/eeg_stanford_stimuli_ids.txt') as ids_txt:
            raw_lines = [l.strip().split('\t') for l in ids_txt.readlines()]
        lines = [l for l in raw_lines]
        words_and_cats = {l[1] : (l[2], 'Unknown') for l in lines}
        #mapping_dictionary = {'Object' : 'Physical object', 'Japan' : 'Physical object'}
        #words_and_cats = {k : mapping_dictionary[v] if v in mapping_dictionary.keys() else v for k, v in words_and_cats.items()}

        return words_and_cats

    def mitchell(self):
        with open('resources/mitchell_words_and_cats.txt') as mitchell_file:
            raw_lines = [l.strip().split('\t') for l in mitchell_file.readlines()]
            words_and_cats = {l[0].capitalize() : l[1].capitalize() for l in raw_lines}
            mapping_dictionary = {'Manmade' : 'Physical object', 'Buildpart' : '', 'Bodypart' : 'Human body'}
            words_and_cats = {k : (mapping_dictionary[v], 'Unknown') if v in mapping_dictionary.keys() else v for k, v in words_and_cats.items()}
        return words_and_cats

    def full_wiki(self):

        words_and_cats = collections.defaultdict(tuple)

        for root, direct, files in os.walk(os.path.join(self.base_folder, 'wikipedia_entities_list')):
            for f in files:
                with open(os.path.join(self.base_folder, root, f), 'r') as entities_file:
                    all_lines = [l.strip().split('\t') for l in entities_file.readlines() if '\tPlace' in l or '\tPerson' in l]
                for l in all_lines:
                    #person_words.append(l[0]) if l[1] == 'Person' else places_words.append(l[0])
                    name = re.sub('_', ' ', l[0])
                    if len(l) > 3:
                        try:
                            coarse = l[2]
                            mapping = {'Fictional' : 'Character (arts)', 'Neighborhood' : 'Neighbourhood','Sports' : 'Athlete', 'Lake' : 'Body of water', 'Sea' : 'Body of water', 'River' : 'Body of water'}
                            #finer_cat = l[3] if l[3] != 'Sea' and l[3] != 'River' and l[3] != 'Lake' else 'Body of water'
                            fine = l[3] if l[3] not in mapping.keys() else mapping[l[3]]
                            words_and_cats[name] = (fine, coarse)
                        except KeyError:
                            print('Couldn\'t find {}'.format(name))

        return words_and_cats

    def men(self):
        words = []
        with open('resources/men.txt', 'r') as men_file:
            all_lines = [l.split()[:2] for l in men_file.readlines()]
        all_words = [final_w for final_w in {re.sub('_.', '', w) : '' for l in all_lines for w in l}.keys()]
        men_words = {w : ('Unknown', 'Unknown') for w in all_words}
        return men_words

    def stopwords(self):
        words = []
        with open('resources/stopwords.txt', 'r') as stopwords_file:
            all_words = [l.strip('\n' ) for l in stopwords_file.readlines()[1:] if len(l) >= 5]
        stopwords = {w : ('Unknown', 'Unknown') for w in all_words}
        return men_words