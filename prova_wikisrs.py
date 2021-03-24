from extract_word_lists import Entities

a = Entities('wikisrs').words
b = Entities('full_wiki').words

w_list = list()
for w in a.keys():
    if w not in b.keys():
        w_list.append(w)

import pdb; pdb.set_trace()
