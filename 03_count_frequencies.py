import argparse
import math
import os

from tqdm import tqdm

from utils import load_vec_files, read_args, read_sentences_folder

args = read_args()

#vecs_file, rankings_folder = load_vec_files(args, computational_model)

### extracting them if not available
all_sentences = dict()

if args.corpus == 'joint':
    for corpus in ['wikipedia', 'opensubtitles']:
        sent_folder = read_sentences_folder(args).replace(args.corpus, corpus)
        sent_file = os.path.join(sent_folder,'{}.sentences'.format(args.corpus_portion)) 
        with open(sent_file) as i:
            with tqdm() as counter:
                for l in i:
                    name_and_sent = l.strip().split('\t')
                    try:
                        all_sentences[name_and_sent[0]].append(name_and_sent[1])
                        counter.update(1)
                    except KeyError:
                        all_sentences[name_and_sent[0]] = [name_and_sent[1]]
else:
    sent_folder = read_sentences_folder(args)
    sent_file = os.path.join(sent_folder,'{}.sentences'.format(args.corpus_portion)) 
    with open(sent_file) as i:
        with tqdm() as counter:
            for l in i:
                name_and_sent = l.strip().split('\t')
                try:
                    all_sentences[name_and_sent[0]].append(name_and_sent[1])
                    counter.update(1)
                except KeyError:
                    all_sentences[name_and_sent[0]] = [name_and_sent[1]]

frequencies = {k : len(v) for k, v in all_sentences.items()}

out_folder = os.path.join('ratings')
os.makedirs(out_folder, exist_ok=True)

for data_mode in ['frequency', 'log_frequency']:
    out_file = os.path.join(
                            out_folder, 
                            'exp_{}_{}_{}_{}_{}_ratings.tsv'.format(
                                       args.experiment_id, 
                                       data_mode, 
                                       args.corpus,
                                       args.corpus_portion,
                                       args.language, 
                                       )
                            )
    with open(out_file, 'w') as o:
        o.write('entity\t{}\n'.format(data_mode))
        for k, v in frequencies.items():
            if data_mode == 'log_frequency':
                val = math.log(v)
            else:
                val = v
            o.write('{}\t{}\n'.format(k, val))
