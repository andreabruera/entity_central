import os
import collections
import re

from tqdm import tqdm

for root, direct, files in os.walk('/import/cogsci/andrea/dataset/wikiextractor_dumps'):
    for f in tqdm(files):
        current_file = collections.defaultdict(list)
        with open(os.path.join(root, f)) as input_file:
            raw_lines = [l.strip() for i, l in enumerate(input_file.readlines())]
        lines_clean = [l for l in raw_lines if l != '' and l != '</doc>']
        for i, l in enumerate(lines_clean):
            if '<doc id="' in l:
                current_key = lines_clean[i+1]
                if len(current_key) > 50:
                    current_key = current_key[:50]
            elif l == current_key:
                pass
            else:
                current_file[current_key].append(l)
        for name, lines in current_file.items():
            file_name = re.sub(' |\/|\.\.\.|\.\.', '_', name)
            initials = name[:2]
            folder_path = '/import/cogsci/andrea/dataset/wikipedia_article_by_article/{}'.format(initials)
            os.makedirs(folder_path, exist_ok=True)
            with open(os.path.join(folder_path, '{}.txt'.format(file_name)), 'w') as output_file:
                for l in lines:
                    output_file.write('{}\n'.format(l))

