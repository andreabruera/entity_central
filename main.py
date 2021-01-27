from extract_word_lists import Entities
from clean_sentences import read_sentences
from extract_bert import bert

#for mode in ['full_wiki', 'eeg_stanford', 'wakeman_henson', 'men', 'stopwords']:
for mode in ['full_wiki']:
    print('Currently extracting vectors for: {}'.format(mode))
    ent_dict = Entities(mode).word_categories
    ent_sentences = read_sentences(ent_dict)
    bert(ent_sentences)
