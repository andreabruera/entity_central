import numpy
import os

def read_datasets():
    with open(os.path.join(
                           'original_ratings', 
                           'anew_montefinese.csv'
                           ), 
                                encoding='latin-1') as i:
        lines = [l.strip().split(',') for l in i.readlines()]
    header = lines[0]
    data = lines[1:]
    word_title = 'Ita_Word'
    word_idx = header.index(word_title)
    variables = {
            'concreteness' : 'M_Con', 
            'familiarity' : 'M_Fam', 
            'imageability' : 'M_Ima',
            'valence' : 'M_Val',
            'arousal' : 'M_Aro',
            'dominance' : 'M_Dom'
            }
    dataset = {k : dict() for k in variables.keys()}
    for variable, col_title in variables.items():
        col_idx = header.index(col_title)
        for l in data:
            if len(l[word_idx]) > 0:
                dataset[variable][l[word_idx]] = float(l[col_idx])

    ### overall affective like Liuzzi et al. 2022
    variable = 'affective'
    variables = [
                  'valence',
                  'arousal',
                  'dominance',
                  ]
    dataset[variable] = dict()
    for l in data:
        if len(l[word_idx]) > 0:
            dataset[variable][l[word_idx]] = list()
            for col_title in variables:
                dataset[variable][l[word_idx]].append(dataset[col_title][l[word_idx]])
    for w, perc_scores in dataset[variable].items():
        assert len(perc_scores) == len(variables)

    ### perceptual
    with open(os.path.join(
                           'original_ratings', 
                           'anew_perceptual.txt'
                           ), 
                              encoding='latin-1') as i:
        lines = [l.strip().replace('"', '').split(' ') for l in i.readlines()]
    header = lines[0]
    data = lines[1:]
    word_title = 'Ita_Word'
    word_idx = header.index(word_title)
    variables = [
            "Auditory",
            "Gustatory", 
            "Haptic",
            "Olfactory",
            "Visual"
            ]
    variable = 'perceptual'
    dataset[variable] = dict()
    for l in data:
        if len(l[word_idx]) > 0:
            dataset[variable][l[word_idx]] = list()
            for col_title in variables:
                col_idx = header.index(col_title)
                dataset[variable][l[word_idx]].append(float(l[col_idx]))
    for w, perc_scores in dataset[variable].items():
        assert len(perc_scores) == len(variables)
    
    return dataset

def read_personally_familiar_entities():
    sub_pers_fam = dict()
    for sub in range(1, 34):
        path = os.path.join(
                            '/', 'import', 'cogsci',
                            'andrea', 'dataset', 'neuroscience',
                            'family_lexicon_eeg', 'derivatives', 
                            'sub-{:02}'.format(sub), 
                            'sub-{:02}_task-namereadingimagery_events.tsv'.format(sub)
                            )
        assert os.path.exists(path)
        with open(path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        key_and_stim = {l[2].strip() : int(l[3]) for l in lines if int(l[3])<50}
        sub_pers_fam[sub] = [k[0] for k in sorted(key_and_stim.items(), key=lambda item : item[1])]
        assert len(sub_pers_fam[sub]) == 16
    return sub_pers_fam

def read_personally_familiar_sentences():

    pers_fam_sentences = dict()
    pers_fam_names = read_personally_familiar_entities()
    for sub, names in pers_fam_names.items():
        pers_fam_sentences[sub] = dict()
        path = os.path.join(
                    'latest_transcriptions',
                    'sub-{:02}_task-namereadingimagery_transcript-automatic.txt'.format(sub)
                    )
        with open(path) as i:
            lines = [l.strip() for l in i.readlines() if len(l)>1]
        assert len(lines) == 16
        for name, line in zip(names, lines):
            pers_fam_sentences[sub][name] = line.strip().split('.')
            assert len(pers_fam_sentences[sub][name]) > 7
            #print(len(pers_fam_sentences[sub][name]))
    return pers_fam_sentences

def read_exp_two_data(sub_marker):

    sub_pers_fam = dict()
    path = os.path.join(
                        '/', 'import', 'cogsci',
                        'andrea', 'dataset', 'neuroscience',
                        'family_lexicon_eeg', 'derivatives', 
                        'sub-{}'.format(sub_marker.replace('_', '')), 
                        'sub-{}task-namereadingimagery_events.tsv'.format(sub_marker)
                        )
    assert os.path.exists(path)
    with open(path) as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]
    #key_and_stim = {l[2].strip() : int(l[3]) for l in lines if int(l[3])<50}
    #sub_pers_fam[sub] = [k[0] for k in sorted(key_and_stim.items(), key=lambda item : item[1])]
    #assert len(sub_pers_fam[sub]) == 16
    reaction_times = dict()
    coarse = dict()
    fame = dict()
    for l in lines:
        if l[7].strip() == 'familiar':
            key = '{}{}'.format(sub_marker, l[2].strip())
        else:
            key = l[2].strip()
        try:
            reaction_times[key].append(float(l[4]))
        except KeyError:
            reaction_times[key] = [float(l[4])]
        coarse[key] = l[6]
        fame[key] = l[7]
    reaction_times = {k : numpy.average(v) for k, v in reaction_times.items()}
    rts_only = [reaction_times[k] for k in sorted(reaction_times.keys())]
    one_zero_rts = [(v-min(rts_only))/(max(rts_only)/min(rts_only)) for v in rts_only]
    one_zero_reaction_times = {k : numpy.array(v) for v, k in zip(one_zero_rts, sorted(reaction_times.keys()))}

    return one_zero_reaction_times, coarse, fame
