from extract_word_lists import Entities
from clean_sentences import read_sentences
from extract_bert import bert

#for mode in ['full_wiki', 'eeg_stanford', 'wakeman_henson', 'men', 'stopwords']:
for mode in ['stopwords']:
    print('Currently extracting vectors for: {}'.format(mode))
    ent_dict = Entities(mode).words
    if mode == 'full_wiki' or mode == 'wakeman_henson':
        ent_sentences = read_sentences(ent_dict)
    else:
        ent_sentences = read_sentences(ent_dict, all_common_nouns=True)
    bert(ent_sentences)
