import argparse
import math
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--sentences_file',
                    required=True,
                    )
parser.add_argument(
                    '--output_identifier',
                    required=True,
                    help='how to name the resulting frequencies file'
                    )
args = parser.parse_args()

frequencies = dict()

with open(args.sentences_file) as i:
    with tqdm() as counter:
        for l in i:
            name = l.strip().split('\t')[0]
            try:
                frequencies[name] += 1
                counter.update(1)
            except KeyError:
                frequencies[name] = 1

with open(os.path.join('frequencies', '{}.freqs'.format(args.output_identifier)), 'w') as o:
    o.write('entity\tfrequency\tlog_frequency\n')
    for k, v in frequencies.items():
        o.write('{}\t{}\t{}\n'.format(k, v, math.log(v)))
