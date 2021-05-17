import os
import re
import collections
import numpy

class Entities:

    def __init__(self, required_words):

        self.base_folder = '/import/cogsci/andrea/github/names_count'
        self.word_set = required_words

        if self.word_set == 'wikisrs':
            self.words = self.wikisrs()
        elif self.word_set == 'wakeman_henson':
            self.words = self.wakeman_henson()
        elif self.word_set == 'full_wiki':
            self.words = self.full_wiki()
        elif self.word_set == 'men':
            self.words = self.men()
        elif self.word_set == 'eeg_stanford':
            self.words = self.eeg_stanford()
        elif self.word_set == 'mitchell':
            self.words = self.mitchell()
        elif self.word_set == 'eeg_one':
            self.words = self.eeg_one()
        elif self.word_set == 'stopwords':
            self.words = self.stopwords()

        self.word_categories = self.expand_word_categories()

    def expand_word_categories(self):

        if self.word_set == 'full_wiki' or self.word_set == 'wakeman_henson' or self.word_set == 'wikisrs':
            entities_list = {ent : 'ent' for ent in self.words.keys()}

        elif self.word_set == 'eeg_one':
            entities_list = {ent : 'ent' for ent in self.words.keys()}
            cats = ['Musician', 'Actor', 'Politician', 'Writer', \
                    'Body of water', 'City', 'Monument', 'Country']
            for c in cats:
                entities_list[c] = 'cat' 
    
        else:
            entities_list = {ent : 'cat' for ent in self.words.keys() if re.sub('[0-9]', '', ent) != ''}

        fine_list = {cat[0] : 'cat' for ent, cat in self.words.items() if cat[0] != 'Unknown'}
        coarse_list = {cat[1] : 'cat' for ent, cat in self.words.items() if cat[0] != 'Unknown'}

        expanded_word_categories = {**entities_list, **fine_list, **coarse_list}

        return expanded_word_categories

    def wikisrs(self):

        wikisrs_mapping = {'Fyodor Dostoyevsky' : 'Fyodor Dostoevsky', 
                           'Ken Griffey, Jr.' : 'Ken Griffey Jr.',
                           'Martin van Buren' : 'Martin Van Buren',
                           'Libertarian Party' : 'Libertarian_Party_(United_States).txt',
                           'Green Party' : 'Green_Party_of_the_United_States.txt', 
                           'Technical University Munich' : 'Technical_University_of_Munich.txt',
                          }
        with open('resources/WikiSRS/WikiSRS_similarity.csv') as input_file:
            lines = [l.strip().split('\t')[2:5] for l in input_file.readlines()][1:]
        words_sim = list(set([l[i] for l in lines for i in range(2)]))
        with open('resources/WikiSRS/WikiSRS_relatedness.csv') as input_file:
            lines = [l.strip().split('\t')[2:5] for l in input_file.readlines()][1:]
        words_rel = list(set([l[i] for l in lines for i in range(2)]))
        words_final = set(words_rel + words_sim)
        names_and_cats = {w : ('Unknown', 'Unknown') for w in words_final}

        return names_and_cats
       
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

    def eeg_one(self):

        with open('resources/exp_one_stimuli.txt') as ids_txt:
            lines = [l.strip().split('\t')[1:] for l in ids_txt.readlines()][1:]
        with open('resources/final_words_exp_one.txt') as ids_txt:
            actual_words = [l.strip() for l in ids_txt.readlines()]
        coarse_mapper = {'persona' : 'Person', 'luogo' : 'Place'}
       
        fine_mapper = {'musicista' : 'Musician', 'attore' : 'Actor', \
                       'politico' : 'Politician', 'scrittore' : 'Writer', \
                       'corso d\'acqua' : 'Body of water', 'città' : 'City', \
                       'monumento' : 'Monument', 'stato' : 'Country'}

        names_mapper = {'Papa Francesco' : 'Pope Francis', 'Gandhi' : 'Mahatma Gandhi', \
                        'Martin Luther King' : 'Martin Luther King Jr.', \
                        'J.K. Rowling' : 'J. K. Rowling', 'Lev Tolstoj' : 'Leo Tolstoy', \
                        'Marylin Monroe' : 'Marilyn Monroe', 'Germania' : 'Germany',\
                        'Israele' : 'Israel', 'Giappone' : 'Japan', 'Svizzera' : 'Switzerland', \
                        'Regno Unito' : 'United Kingdom', 'Spagna' : 'Spain', \
                        'Sud Africa' : 'South Africa', 'Sud Corea' : 'South Korea', \
                        'Pechino' : 'Beijing', 'Rio De Janeiro' : 'Rio de Janeiro', \
                        'Atene' : 'Athens', 'Oceano Pacifico' : 'Pacific Ocean', \
                        'Canale della Manica' : 'English Channel', \
                        'Mar Mediterraneo' : 'Mediterranean Sea', 'Mar dei Caraibi' : 'Caribbean Sea', \
                        'Mar Nero' : 'Black Sea', 'Nilo' : 'Nile', 'Parigi' : 'Paris',\
                        'Mare del Nord' : 'North Sea', 'Baia di Hudson' : 'Hudson Bay', \
                        'Oceano Atlantico' : 'Atlantic Ocean', 'Roma' : 'Rome',\
                        'Monte Rushmore' : 'Mount Rushmore', 'Piramidi di Giza' : 'Giza pyramid complex', \
                        'Muraglia Cinese' : 'Great Wall of China', 'Mosca' : 'Moscow',\
                        'Macchu Picchu' : 'Machu Picchu', 'Monte Saint-Michel' : 'Mont-Saint-Michel', \
                        'Torre di Pisa' : 'Leaning Tower of Pisa', 'Mare Rosso' : 'Red Sea',\
                        'Sagrada Familia' : 'Sagrada_Família', 'New York' : 'New York City',\
                        'Madonna' : 'Madonna (entertainer)'}

        words_and_cats = dict()
        for l in lines:
            if l[2] in fine_mapper.keys() and l[0] in actual_words:
                
                if l[0] in names_mapper.keys():
                    name = names_mapper[l[0]]
                else:
                    name = l[0]
                words_and_cats[name] = (fine_mapper[l[2]], coarse_mapper[l[1]])

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
                            mapping = {'Fictional' : 'Character (arts)', 'Neighborhood' : 'Neighbourhood','Sports' : 'Athlete', 'Lake' : 'Body of water', 'Sea' : 'Body of water', 'River' : 'Body of water', 'Music' : 'Musician', 'Literature' : 'Writer', 'Politics' : 'Politician', 'Film' : 'Actor', 'Sports' : 'Athlete'}
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
        men_words = {w.capitalize() : ('Unknown', 'Unknown') for w in all_words}
        return men_words

    def stopwords(self):
        words = []
        with open('resources/stopwords.txt', 'r') as stopwords_file:
            all_words = [l.strip('\n' ) for l in stopwords_file.readlines()[1:] if len(l) >= 5]
        stopwords = {w.capitalize() : ('Unknown', 'Unknown') for w in all_words}
        return stopwords
