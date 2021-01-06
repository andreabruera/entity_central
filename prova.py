from extract_word_lists import Entities
from write_vectors_to_file import read_sentences
from extract_bert import bert

ent_dict = Entities('full_wiki').words
ent_sentences = read_sentences(ent_dict)
bert(ent_sentences)
