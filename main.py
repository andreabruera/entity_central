from extract_word_lists import Entities
from clean_sentences import read_sentences
from extract_bert import bert

#for mode in ['full_wiki', 'eeg_stanford', 'wakeman_henson', 'men', 'stopwords']:
for mode in ['wikisrs']:
    print('Currently extracting vectors for: {}'.format(mode))
    ent_dict = Entities(mode).word_categories

    if mode == 'wikisrs':
        b = Entities('full_wiki').words
        w_list = list()
        for w in ent_dict.keys():
            if w not in b.keys():
                w_list.append(w)

        ent_dict = {k : 'ent' for k in w_list}

    ent_sentences = read_sentences(ent_dict)
    bert(ent_sentences)
