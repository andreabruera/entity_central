import argparse
import numpy
import os
import random
import re

from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm

def layers(args, n_layers):
    if args.layer == 'low_four':
        layer_start = 1
        ### outputs has at dimension 0 the final output
        layer_end = 5
    if args.layer == 'low_six':
        layer_start = 1
        ### outputs has at dimension 0 the final output
        layer_end = 7
    if args.layer == 'low_twelve':
        layer_start = 1
        ### outputs has at dimension 0 the final output
        layer_end = 13
    if args.layer == 'mid_four':
        layer_start = int(n_layers/2)-2
        layer_end = int(n_layers/2)+3
    if args.layer == 'mid_six':
        layer_start = int(n_layers/2)-3
        layer_end = int(n_layers/2)+4
    if args.layer == 'mid_twelve':
        layer_start = int(n_layers/2)-6
        layer_end = int(n_layers/2)+7
    if args.layer == 'upper_four':
        layer_start = int(n_layers/2)
        layer_end = int(n_layers/2)+5
    if args.layer == 'upper_six':
        layer_start = int(n_layers/2)
        layer_end = int(n_layers/2)+7
    if args.layer == 'cotoletta_eight':
        layer_start = int(n_layers/2)+int(int(n_layers/2)/2)-4
        layer_end = int(n_layers/2)+int(int(n_layers/2)/2)+5
    if args.layer == 'cotoletta_six':
        layer_start = int(n_layers/2)+int(int(n_layers/2)/2)-3
        layer_end = int(n_layers/2)+int(int(n_layers/2)/2)+4
    if args.layer == 'cotoletta_four':
        layer_start = int(n_layers/2)+int(int(n_layers/2)/2)-2
        layer_end = int(n_layers/2)+int(int(n_layers/2)/2)+3
    if args.layer == 'top_four':
        layer_start = -4
        ### outputs has at dimension 0 the final output
        layer_end = n_layers+1
    if args.layer == 'top_six':
        layer_start = -6
        ### outputs has at dimension 0 the final output
        layer_end = n_layers+1
    if args.layer == 'top_two':
        layer_start = -2
        ### outputs has at dimension 0 the final output
        layer_end = n_layers+1
    if args.layer == 'top_one':
        layer_start = -1
        ### outputs has at dimension 0 the final output
        layer_end = n_layers+1
    if args.layer == 'top_twelve':
        layer_start = -12
        ### outputs has at dimension 0 the final output
        layer_end = n_layers+1
    if args.layer == 'top_eight':
        layer_start = -8
        layer_end = n_layers+1
    return layer_start, layer_end

def read_sentences_folder(args):

    out_folder = os.path.join(
                              'sentences', 
                              args.corpus, 
                              args.language, 
                              args.experiment_id
                              )
    os.makedirs(out_folder, exist_ok=True)

    return out_folder

def return_entity_file(entity):
    entity = '{}{}'.format(entity[0].capitalize(), entity[1:])
    wiki_file = '{}.txt'.format(re.sub('[^a-zA-Z0-9]', '_', entity))
    return wiki_file

def load_comp_model_name(args):
    if args.model == 'ITGPT2medium':
        model_name = 'GroNLP/gpt2-medium-italian-embeddings'
        computational_model = 'ITGPT2'
        out_shape = (1024, )
    if args.model == 'llama-65b':
        model_name = 'decapoda-research/llama-65b-hf'
        computational_model = 'llama'
        out_shape = (1024, )
    if args.model == 'gpt2-large':
        model_name = 'gpt2-large'
        computational_model = 'gpt2-large'
        out_shape = (1280, )
    if args.model == 'MBERT':
        model_name = 'bert-base-multilingual-cased'
        computational_model = 'MBERT'
        out_shape = (768, )
    if args.model == 'BERT_large':
        model_name = 'bert-large-cased'
        computational_model = 'BERT_large'
        out_shape = (1024, )
    if args.model == 'xlm-roberta-large':
        model_name = 'xlm-roberta-large'
        computational_model = 'xlm-roberta-large'
        out_shape = (1024, )
    if args.model in ['valence', 'concreteness', 'imageability', 'arousal', 'dominance']:
        model_name = '{}'.format(args.model)
        computational_model = '{}'.format(args.model)
        out_shape = (1, )
    if args.model == 'perceptual':
        model_name = '{}'.format(args.model)
        computational_model = '{}'.format(args.model)
        out_shape = (5, )
    if args.model == 'affective':
        model_name = '{}'.format(args.model)
        computational_model = '{}'.format(args.model)
        out_shape = (3, )
    if args.model == 'w2v_sentence':
        model_name = 'w2v_sentence'
        computational_model = 'w2v_sentence'
        out_shape = (300, )
    if args.model == 'wikipedia2vec_sentence':
        model_name = 'wikipedia2vec_sentence'
        computational_model = 'wikipedia2vec_sentence'
        if args.language == 'it':
            out_shape = (300, )
        if args.language == 'en':
            out_shape = (500, )

    return model_name, computational_model, out_shape

def read_args(vector_extraction=False, contextualized_selection=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
    parser.add_argument(
                        '--language', 
                        choices=['it', 'en'], 
                        required=True
                        )
    parser.add_argument(
                        '--corpus_portion',
                        choices=['entity_articles', 'full_corpus', 'entity_sentences'],
                        required=True
                        )
    parser.add_argument('--corpus', choices=['opensubtitles', 'wikipedia', 'joint'], required=True)
    parser.add_argument('--layer', choices=[
                                            'low_four',
                                            'mid_four', 
                                            'top_four',
                                            'low_six',
                                            'mid_six', 
                                            'top_six',
                                            'top_eight',
                                            'low_twelve',
                                            'mid_twelve',
                                            'top_twelve', 
                                            'top_two', 
                                            'top_one', 
                                            'upper_four',
                                            'upper_six',
                                            'cotoletta_four',
                                            'cotoletta_six',
                                            'cotoletta_eight',
                                            'jat_mitchell',
                                            ],
                        required=True, help='Which layer?')
    parser.add_argument('--model', choices=[
                                            'BERT_large',
                                            'MBERT', 
                                            'ITGPT2medium',
                                            'xlm-roberta-large',
                                            'gpt2-large',
                                            'w2v_sentence',
                                            'wikipedia2vec_sentence',
                                            'valence',
                                            'arousal',
                                            'concreteness',
                                            'imageability',
                                            'perceptual',
                                            'affective',
                                            ],
                        required=True, help='Which model?')
    parser.add_argument('--debugging', action='store_true')

    ### only required when extracting vectors
    if vector_extraction:
        parser.add_argument(
                            '--cuda', 
                            choices=['0', '1', '2',
                                               ],
                            required=True, 
                            help='Which cuda device?'
                            )
        parser.add_argument(
                            '--words_used',
                            choices=[
                                     'all', 
                                     'content',
                                     ],
                            default='content_words',
                            help='Averaging all words or just content words?'
                            )
        parser.add_argument(
                            '--amount_vectors', 
                            choices=[
                                     'all', 
                                     'ten', 
                                     'hundred'
                                     ],
                            default='all',
                            help='How many vectors to use?',
                            )

    ### only required when selecting best vectors
    if contextualized_selection:
        parser.add_argument('--static_sentences', action='store_true')
        parser.add_argument('--one', action='store_true')
        #parser.add_argument('--average', choices=[-12, 24], type=int, required=True)
        if contextualized_selection:

            times_and_labels = [
                    '0-200ms',
                    '200-300ms',
                    '300-500ms',
                    '500-800ms',
                    ]
            #parser.add_argument('--time_window', choices=times_and_labels, 
            #                    required=True)
        parser.add_argument('--individuals', action='store_true')
        parser.add_argument('--predicted', action='store_true')
        parser.add_argument('--random', action='store_true')

    args = parser.parse_args()
    

    return args

def read_full_wiki_vectors(vecs_file, out_shape):
    with open(vecs_file, 'r') as o:
        lines = [l.strip().split('\t') for l in o.readlines()]
    entity_vectors = dict()
    all_sentences = dict()
    counter = 0
    marker = False
    for l in lines:
        vec = numpy.array(l[2:], dtype=numpy.float64)
        try:
            assert vec.shape == out_shape
        except AssertionError:
            marker = True
            continue
        try:
            assert numpy.sum(vec) != 0.
        except AssertionError:
            marker = True
            continue
        try:
            entity_vectors[l[0]].append((counter, vec))
            all_sentences[l[0]].append((counter, l[1]))
        except KeyError:
            entity_vectors[l[0]] = [(counter, vec)]
            all_sentences[l[0]] = [(counter, l[1])]
        counter += 1
    if marker:
        print('careful, some sentences had the wrong dimensionality!')

    return entity_vectors, all_sentences

def load_vec_files(args, computational_model):

    os.makedirs(
                'replication_models', 
                exist_ok=True
               )

    repl_file = os.path.join(
                             'replication_models',
                             'exp_{}_{}_{}_{}_{}_words_{}_vectors.tsv'.format(
                                      args.experiment_id, 
                                      args.model,
                                      args.language, 
                                      args.corpus_portion,
                                      args.words_used,
                                      args.amount_vectors
                                      )
                                    )




    base_folder = os.path.join(
                              'contextualized_vector_selection', 
                              args.corpus,
                              args.corpus_portion,
                              args.language,
                              args.experiment_id,
                              computational_model,
                              args.layer,
                              )
    vecs_folder = os.path.join(
                              base_folder,
                              'phrase_vectors',
                              )

    os.makedirs(vecs_folder, exist_ok=True)
    vecs_file= os.path.join(
                            vecs_folder, 
                            'all.vectors'
                            )
    if args.individuals or args.random or args.one:
        rankings_folder = os.path.join(
                                  base_folder,
                                  'rankings',
                                  'individuals',
                                  )

        os.makedirs(rankings_folder, exist_ok=True)
    else:
        rankings_folder = os.path.join(
                                  base_folder,
                                  'rankings',
                                  #'average_{}'.format(args.average),
                                  #args.time_window,
                                  )

        os.makedirs(rankings_folder, exist_ok=True)

    return vecs_file, rankings_folder, repl_file

def read_brain_data(args):
    ### now ranking sentences for each subject!
    ### Loading brain data
    brain_data_file = os.path.join(
                                   'brain_data', 
                                   args.experiment_id, 
                                   'rough_and_ready_exp_{}_average_24_300-500ms.eeg'.format(
                                                                 args.experiment_id, 
                                                                 #args.average, 
                                                                 #args.time_window
                                                                 )
                                   )
    assert os.path.exists(brain_data_file)
    with open(brain_data_file) as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    subjects = set([int(l[0]) for l in lines])
    brain_data = {sub : {l[1] : numpy.array(l[2:], dtype=numpy.float64) for l in lines if int(l[0])==sub} for sub in subjects}

    return brain_data

def read_entity_sentences(args):
    ### reading entity list
    entity_list_file = os.path.join(
                                    'brain_data',  
                                    args.experiment_id, 
                                    'wikidata_ids_{}.txt'.format(args.experiment_id)
                                    )
    assert os.path.exists(entity_list_file)
    with open(entity_list_file) as i:
        ent_lines = [l.strip().split('\t') for l in i.readlines()]
    for l_i, l in enumerate(ent_lines):
        if l_i == 0:
            #print(l)
            assert l[0] == 'entity'
            assert l[1] == 'wikidata_id'
        assert len(l) == 2
    entities = {l[0] : l[1] for l in ent_lines[1:] if l[0][0]!='#'}

    aliases = dict()
    files = dict()
    ### creating output folder


    if args.corpus_portion == 'entity_sentences':
        if args.language == 'it':
            corpus_folder = '/import/cogsci/andrea/dataset/corpora/wexea_it/articles_2/'
        elif args.language == 'en':
            corpus_folder = '/import/cogsci/andrea/dataset/corpora/wexea_annotated_wiki/ready_corpus/original_articles/'
        ### checking wiki files
        marker = False
        for k, wikidata_id in tqdm(entities.items()):
            ent_dict = get_entity_dict_from_api(wikidata_id)
            main_alias = ent_dict['labels'][args.language]['value']
            #print(main_alias)
            file_k = return_entity_file(main_alias)
            if main_alias == 'J. K. Rowling':
                file_k = 'J._K._Rowling.txt'
                main_alias = 'J.K. Rowling'
            if main_alias == 'Sagrada Família':
                file_k = 'Sagrada_Família.txt'
                main_alias = 'Sagrada Familia'
            sub_folder = re.sub("[^A-Za-z0-9]", '_', main_alias).lower()
            if args.language == 'it':
                if main_alias == 'città':
                    file_k = 'Città.txt'
                    #main_alias = 'Città'
                if main_alias == "massa d'acqua":
                    file_k = "Massa_d'acqua.txt"
                    #main_alias = 'Sagrada Familia'
                if main_alias == "Madonna":
                    file_k = "Madonna_(cantante).txt"

                current_folder = os.path.join(corpus_folder, sub_folder[:1], sub_folder[:2], sub_folder[:3])
            else:
                if main_alias == "Madonna":
                    file_k = "Madonna_(entertainer).txt"
                current_folder = os.path.join(corpus_folder, sub_folder[:3])

                #print(file_k)
                #print(current_folder)
            assert os.path.exists(current_folder)
            file_path = os.path.join(current_folder, file_k)
            try:
                #print(file_path)
                assert os.path.exists(file_path)
                #shutil.copy(file_path, articles_path)
                with open(file_path) as i:
                    lines = [l.strip() for l in i.readlines()]
                assert len(lines) > 0
                files[k] = lines
            except AssertionError:
                print(main_alias)
                print([k, file_k])
                print(file_path)
                #import pdb; pdb.set_trace()
                marker = True
            aliases[k] = [main_alias]
            if len(k.split()) > 1:
                aliases[k].append(k.split()[-1])
            if args.language in ent_dict['aliases'].keys():
                for al in ent_dict['aliases'][args.language]:
                    aliases[k].append(al['value'])
        aliases = {k : sorted(v, key=lambda item : len(item), reverse=True) for k, v in aliases.items()}
        all_sentences = {k : list() for k in aliases.keys()}
        if marker:
            raise RuntimeError('Some names are not correctly written!')
    else:
        corpus_folder = read_sentences_folder(args)
        all_sentences = dict()
        with open(os.path.join(corpus_folder, '{}.sentences'.format(args.corpus_portion))) as i:
            for l in i.readlines():
                k = l.strip().split('\t')[0]
                line = l.strip().split('\t')[1]
                if k not in all_sentences.keys():
                    all_sentences[k] = [line]
                else:
                    all_sentences[k].append(line)
        max_n = 1000
        all_sentences = {k : random.sample(v, k=min(max_n, len(v))) for k, v in all_sentences.items()}


    formula_one = '(?<=\[\[)(.+?)\|.+?(?=\]\])'
    formula_two = '(?<=\[\[)(.+?)(?=\]\])'
    for key in tqdm(aliases.keys()):
        lines = files[key]
        for line in lines:
            if args.experiment_id == 'two':
                if args.model != 'w2v_sentence':
                    line = re.sub(formula_one, r'\1', line)
                    line = re.sub('\[|\]', '', line)
            else:
                if args.model not in ['ITGPT2medium', 'MBERT', 'BERT_large', 'xlm-roberta-large', 'gpt2-large']:
                    line = re.sub(formula_one, r'\1', line)
                    line = re.sub('\[|\]', '', line)
            if len(line.split()) > 5:
                all_sentences[key].append(line)

    return all_sentences

def write_vectors(o, current_entity_vectors, s, out_shape):

    for stim, vecz in current_entity_vectors.items():
        print([stim, s, len(vecz)])
        print([stim, s, set([v[0].shape for v in vecz])])
        vec = numpy.nanmean([v[0] for v in vecz], axis=0)
        #for vec, line in vecz:
        #try:
        assert vec.shape == out_shape
        #except AssertionError:
        #    print([stim, line])
        #line = line.replace('\t', ' ')
        o.write('{}\t{}\t'.format(s, stim))
        for dim in vec:
            o.write('{}\t'.format(float(dim)))
        o.write('\n')
