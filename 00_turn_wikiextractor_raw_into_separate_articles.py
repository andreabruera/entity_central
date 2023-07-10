import argparse
import os
import re

from tqdm import tqdm

from utils import return_entity_file

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--wikiextractor_dump_folder', 
                    required=True
                    )
parser.add_argument(
                    '--output_folder', 
                    required=True
                    )
args = parser.parse_args()

assert os.path.exists(args.wikiextractor_dump_folder)
os.makedirs(args.output_folder, exist_ok=True)

with tqdm() as counter:
    for root, direc, files in os.walk(args.wikiextractor_dump_folder):
        for f in files:
            file_path = os.path.join(root, f)
            with open(file_path, errors='ignore', encoding='utf-8') as i:
                docs = i.read().split('</doc>\n')
                #lines = ''.join([l for l in i.readlines()])
            #lines = [l for l in lines.split('<doc id') if len(l)>1]
            for doc in docs:
                entity = re.findall('<doc.+?title="(.+?)">', doc)
                try:
                    assert len(entity) == 1
                except AssertionError:
                    print(entity)
                    print(doc)
                    continue
                entity = entity[0]
                txt = re.sub('<doc.+?title="(.+?)">', '', doc).strip()
                txt = re.sub(r'\n+', r'\n', txt)
                if len(txt.split('\n')) > 1:
                    wiki_file = return_entity_file(entity) 
                    wiki_file = os.path.join(
                                             args.output_folder,
                                             wiki_file
                                             )
                    with open(wiki_file, 'w') as o:
                        o.write(txt)
                    counter.update(1)
