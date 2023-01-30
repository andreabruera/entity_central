import argparse
import multiprocessing
import os
import re
#import spacy

from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm

from utils import read_sentences_folder, return_entity_file

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--entity_list_file',
                    required=True,
                    )
parser.add_argument('--corpus', choices=['opensubtitles', 'wikipedia'], required=True)
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
parser.add_argument(
                    '--corpus_portion',
                    choices=['entity_articles', 'full_corpus'],
                    required=True
                    )
parser.add_argument(
                    '--debugging',
                    action='store_true'
                    )
args = parser.parse_args()

def sensible_finder(all_args):

    lines = all_args[0]
    stimuli = all_args[1]
    args = all_args[2]

    if args.corpus == 'opensubtitles':

        new_lines = list()
        new_line = ''
        for l in lines:
            if len(new_line.split()) > 60:
                new_lines.append(new_line)
                new_line = ''
            else:
                new_line = '{} {}'.format(new_line, l)
        new_lines.append(new_line)
        del lines
        lines = new_lines.copy()

    ent_sentences = {s : list() for s in stimuli}

    with tqdm() as counter:
        for l in lines:
            for stimulus, aliases in stimuli.items():
                formula = '(?<!\w){}(?!\w)'.format(stimulus)
                for alias in aliases:
                    formula = '{}|(?<!\w){}(?!\w)'.format(formula, alias)
                new_l = re.sub(r'{}'.format(formula),r'[SEP] {} [SEP]'.format(aliases[0]), l)

                if '[SEP]' in new_l:
                    ent_sentences[stimulus].append(new_l)
                    counter.update(1)

    return ent_sentences

### reading entity list
assert os.path.exists(args.entity_list_file)
with open(args.entity_list_file) as i:
    ent_lines = [l.strip().split('\t') for l in i.readlines()]
for l_i, l in enumerate(ent_lines):
    if l_i == 0:
        assert l[0] == 'entity'
        assert l[1] == 'wikidata_id'
    assert len(l) == 2
entities = {l[0] : l[1] for l in ent_lines[1:] if l[0][0]!='#'}

aliases = dict()
files = dict()
### creating output folder

out_folder = read_sentences_folder(args)

all_sentences = {k : list() for k in aliases.keys()}

if args.corpus == 'wikipedia':
    corpus_folder = os.path.join('/', 'import', 'cogsci',
                                 'andrea', 'dataset', 
                                 'corpora', 'wikipedia_italian',
                                 'it_wiki_article_by_article'
                                 )
if args.corpus == 'opensubtitles':
    corpus_folder = os.path.join('/', 'import', 'cogsci',
                                 'andrea', 'dataset', 
                                 'corpora', 
                                 #'opensubs_it_ready'
                                 'OpenSubtitles', 'opensubtitles-parser', 'first_pass'
                                 )

### checking wiki files
marker = False
assert os.path.exists(corpus_folder)
for k, wikidata_id in tqdm(entities.items()):
    ent_dict = get_entity_dict_from_api(wikidata_id)
    main_alias = ent_dict['labels'][args.language]['value']
    #print(main_alias)
    file_k = return_entity_file(main_alias)
    if args.corpus_portion == 'entity_articles':
        file_path = os.path.join(corpus_folder, file_k)
        try:
            assert os.path.exists(file_path)
            with open(file_path) as i:
                lines = [l.strip() for l in i.readlines()]
            assert len(lines) > 0
            files[k] = lines
        except AssertionError:
            print([k, file_k])
            marker = True
    aliases[k] = [main_alias]
    if args.language in ent_dict['aliases'].keys():
        for al in ent_dict['aliases'][args.language]:
            aliases[k].append(al['value'])

if args.corpus_portion == 'entity_articles':
    if marker:
        raise RuntimeError('Some names are not correctly written!')


    for key in tqdm(aliases.keys()):
        ent_sentences = sensible_finder([files[key], aliases])
        for k, v in ent_sentences.items():
            all_sentences[k].extend(v)

elif args.corpus_portion == 'full_corpus':

    print('now loading documents...')
    corpora = list()
    counter = 0
    for f in tqdm(os.listdir(corpus_folder)):
        if args.debugging:
            if counter > 100:
                continue
            else:
                counter += 1
        with open(os.path.join(corpus_folder, f)) as i:
            lines = [l.strip() for l in i.readlines()]
        corpora.append(lines)

    print('now running the actual sentence selection')
    ### Running
    with multiprocessing.Pool(processes=24) as pool:
       results = pool.map(sensible_finder, [[corpus, aliases, args] for corpus in corpora])
       pool.terminate()
       pool.join()

    ### Reorganizing results
    all_sentences = {k : list() for k in aliases.keys()}
    for ent_dict in results:
        for k, v in ent_dict.items():
            all_sentences[k].extend(v)

### Trying to avoid repetitions
final_sents = {k : list(set(v)) for k, v in all_sentences.items()}

### Writing to file
with open(os.path.join(out_folder, '{}.sentences'.format(args.corpus_portion)), 'w') as o:
    for stimulus, ent_sentences in all_sentences.items():
        for sent in ent_sentences:
            clean_sent = sent.replace('\n', ' ').replace('\t', ' ').strip()
            o.write('{}\t{}\n'.format(stimulus, clean_sent))
