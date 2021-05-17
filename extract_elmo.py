import torch
import os
import numpy
import collections
import re

from allennlp.data import Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.modules.elmo import Elmo
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers import WhitespaceTokenizer

from tqdm import tqdm

def elmo(entities_and_sentences_dict, args):

    tokenizer = WhitespaceTokenizer()
    token_indexer = ELMoTokenCharactersIndexer()
    vocab = Vocabulary()

    base_folder = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'elmo')
    elmo_options_file = (os.path.join(base_folder, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'))
    elmo_weight_file = (os.path.join(base_folder, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'))
    elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file, weight_file=elmo_weight_file, \
                                       requires_grad=False, )    
    embedder = BasicTextFieldEmbedder(token_embedders={"elmo_tokens": elmo_embedding})

    elmo_vectors = collections.defaultdict(list)       

    for entity, sentences in tqdm(entities_and_sentences_dict.items()):        
        for sentence in sentences:
            ### Removing BERT's annotation
            if args.extraction_mode == 'full_sentence' or args.extraction_mode == 'unmasked':
                sentence = sentence.replace('[SEP]', ' ')
            elif args.extraction_mode == 'masked':
                sentence = re.sub('\[SEP\].+\[SEP\]', 'CHAR', sentence)
            #sentence = [w for w in sentence if w != '']
            tokens = tokenizer.tokenize(sentence)
            text_field = TextField(tokens, {"elmo_tokens": token_indexer})
            text_field.index(vocab)
            token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
            tensor_dict = text_field.batch_tensors([token_tensor])
            embedded_tokens = embedder(tensor_dict)

            if args.extraction_mode == 'full_sentence':

                elmo_representation = numpy.average(embedded_tokens[0, :, :].detach().numpy(), axis=0)
                relevant_indices = ['', '']

            elif args.extraction_mode == 'masked':

                ### Getting the important indices
                relevant_indices = [i for i, input_token in enumerate(tokens) if 'CHAR' in str(input_token)] 

                mentions = list()
                for i in relevant_indices:
                    mention_representation = embedded_tokens[0, i, :].detach().numpy()
                    mentions.append(mention_representation)
                elmo_representation = numpy.average(mentions, axis=0)

            elif args.extraction_mode == 'unmasked':

                ### Getting the important indices
                name = entity.split() + entity.lower().split()
                relevant_indices = [i for i, input_token in enumerate(tokens) if str(input_token) in name] 

                mentions = list()
                for i in relevant_indices:
                    mention_representation = embedded_tokens[0, i, :].detach().numpy()
                    mentions.append(mention_representation)
                elmo_representation = numpy.average(mentions, axis=0)
            
            if len(relevant_indices) >= 1 and elmo_representation.shape == (1024,):
                elmo_vectors[entity].append((sentence, [elmo_representation]))
            else:
                print('sentence : {}'.format(sentence))
                print('entity: {}'.format(entity))

    return elmo_vectors
