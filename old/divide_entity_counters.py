import os

with open('../resources/thresholded_entities_counter.txt') as input_file:
    lines = [l.strip().split('\t') for l in input_file.readlines()]

output_folder = '/import/cogsci/andrea/github/fame/resources/wikipedia_entities_list'
os.makedirs(output_folder, exist_ok=True)

shortened_lines = [l for l in lines if int(l[1])>500]

counter = 0
name = 0

for l in shortened_lines:
    if counter <= 1000:
        counter += 1
    else:
        name += 1
        counter = 0
    with open(os.path.join(output_folder, 'file_{:02}.txt'.format(name)), 'a') as output_file:
        output_file.write('{}\n'.format('\t'.join(l)))
