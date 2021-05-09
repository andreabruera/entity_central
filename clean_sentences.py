import os
import collections
import re
import nltk
import spacy
import tqdm

from tqdm import tqdm

def read_sentences(entities_dict):

    spacy_model = spacy.load("en_core_web_sm")

    ### Generating paths to each word Wikipedia page file
    word_paths = collections.defaultdict(str)
    print('Now generating the path for the files...')

    c=0
    for current_word, word_type in tqdm(entities_dict.items()):

        file_current_word = re.sub(' ', '_', current_word)
        
        #print([current_word, file_current_word])
        txt_file = '{}.txt'.format(file_current_word)

        if word_type == 'ent':
            short_folder = re.sub('[^a-zA-z0-9]+', '_', current_word)[:3].lower()
            file_name = os.path.join('/import/cogsci/andrea/dataset/corpora/wexea_annotated_wiki/ready_corpus/final_articles', short_folder, txt_file)
        else:
            short_folder = current_word[:2]
            file_name = os.path.join('/import/cogsci/andrea/dataset/corpora/wikipedia_article_by_article', short_folder, txt_file)

        if c<5: ### Just for debugging
            #c+=1
            word_paths[current_word] = file_name
        else:
            break

    ### Reading the list of sentences from the file
    word_wiki_sentences = collections.defaultdict(list)
    not_found = list()
    print('Now reading the raw sentences from files...')

    for current_word, current_path in tqdm(word_paths.items()):

        ### Extracting the list of sentences for the current word
        try:
            with open(current_path) as current_file:
                current_lines = [s for l in current_file.readlines() for s in nltk.tokenize.sent_tokenize(l)]
            word_wiki_sentences[current_word] = current_lines
        except FileNotFoundError:
            not_found.append(current_word)
    if len(not_found) > 0:
        print('Impossible to find the txt files for: {}'.format(not_found))

    ### Cleaning up sentences and making sure to only keep the sentences containing at least one verb
    word_ready_sentences = collections.defaultdict(list)
    print('Now cleaning up the sentences from annotations...')

    for current_word, current_lines in tqdm(word_wiki_sentences.items()):

        current_cat = entities_dict[current_word]

        if current_cat == 'ent':

            current_word = re.sub('_', ' ', current_word)
            current_mention = '[[{}|'.format(current_word)

        else:
            current_mention = re.sub('\(.+$', '', current_word)
            current_mention = re.sub('_', ' ', current_mention).lower()
            if len(current_mention.split()) > 1:
                frequency_finder = {word : len([l.strip() for l in current_lines if '{} '.format(word) in l and word != 'and']) for word in current_mention.split()}
                most_frequent = [w for w, k in sorted(frequency_finder.items(), key= lambda item : item[1], reverse=True)][0]
                current_mention = most_frequent

        selected_lines = [l.strip() for l in current_lines if current_mention in l]
        if len(selected_lines) == 0:
            print('\tmissing sentences for {} - {}'.format(current_word, current_mention))
        ### Removing annotation for other entities than the current one, while surrounding the current one by two '[SEP]' tokens
        for l in selected_lines:

            if current_cat == 'ent':
                cleaned_up_line = ''.join([re.sub('\|.+$', '', piece_two).replace(current_word, '[SEP] {} [SEP]'.format(current_word)) for piece_one in l.split('[[') for piece_two in piece_one.split(']]')])

            else:
                cleaned_up_line = l.replace(current_mention, '[SEP] {} [SEP]'.format(current_mention))

            ### Checking that the line actually contains at least one verb
            spacy_tags = [token.pos_ for token in spacy_model(cleaned_up_line)]
            if 'VERB' in spacy_tags and '[SEP]' in cleaned_up_line:
                word_ready_sentences[current_word].append(cleaned_up_line)
                
    return word_ready_sentences
