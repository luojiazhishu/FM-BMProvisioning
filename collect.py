#!/usr/bin/python
# Use "template.html" in the same folder as template
# to generate a web page containing all the results
# of the Tamarin model.
# Automatically determine whether there are graphs 
# for each lemma.

import sys, os
from functools import cmp_to_key
from jinja2 import Environment, FileSystemLoader

def get_spthy_paths(res_folder, proof_flag = True):
    """ Get paths of .spthy under res_folder
    Para:
        res_folder: relative path to the result folder.
        proof_flag:
            True: use intermediate results under ./proofs/. folder.
            False: use original spthy files under Results/*/. folder.
    Return:
        A list containing paths to .spthy
    """
    proof_folders = []
    with os.scandir(res_folder) as entries:
        for entry in entries:
            if entry.is_dir():
                proof_folders.append(entry.path)

    spthy_list = []
    for folder in proof_folders:
        if proof_flag: folder = folder + '/proofs/'
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file():
                    spthy_list.append(entry.path)

    return spthy_list


def gen_theory(spthy):
    """
    Para:
        spthy: path to a specified .spthy file.
    Return:
        A theory dictionary.
    """
    with open(spthy, 'rb') as f:
        print(spthy)
        # read from end.
        f.seek(-3000,2)
        summary = f.read().decode()
        summary_list = summary.split('\n')

        # locate the results
        index = summary_list.index('summary of summaries:\r')
        # res_list contains the lemmas and results
        res_list = summary_list[index+4:-3]
            
        theory = {'No': 0, 'ClassColor': '',
                'Class': spthy.split('/')[-3],
                'Name': spthy.split('/')[-1],
                'lemmas':{}, 'Time':''}

        theory['ClassColor'] = choose_color(theory['Class'])

        for res in res_list:
            term, res = res.strip().split(':')
            term = term.strip().split(' ')[0]
            if term == 'output':
                pass
            elif term == 'processing':
                theory['Time'] = res.strip()[:8]+res.strip()[-1:]
            else:
                theory['lemmas'][term] = {'color':'', 'result':''}
                theory['lemmas'][term]['result'] = res
                if len(res.strip().split('-')) > 1:
                    theory['lemmas'][term]['color'] = 'red'
                else:
                    theory['lemmas'][term]['color'] = 'green'
                
                graph = os.path.dirname(spthy) + '/src/' + \
                        spthy.split('/')[-1].split('.')[0].replace('*','') \
                        + '/' + term + '.svg'

                if os.path.exists(graph):
                    # relative path to .html
                    relative_graph_path = '/'.join(graph.split('/')[-5:])
                    theory['lemmas'][term]['graph'] = relative_graph_path

    return theory



if __name__ == "__main__":

    try:
        res_folder = sys.argv[1]
    except IndexError:
        res_folder = input("Under which folder are the results saved? (relative to this script)\n")

    Theories = []

    spthy_list = get_spthy_paths(res_folder)
    
    for index, spthy in enumerate(spthy_list):
        theory = gen_theory(spthy)
        theory['No'] = index + 1
        Theories.append(theory)


    # Theories.sort(key=cmp_to_key(lambda x,y:cmp_class(x['Class'],y['Class'])), reverse=True)

    # Use jinja to fill the template
    env = Environment(loader=FileSystemLoader('./'))

    template = env.get_template('template.html')

    with os.scandir(res_folder) as entries:
        res_file = os.path.dirname(next(entries).path) + '/results.html'
        with open(res_file, 'w+') as f: 
            res = template.render(theories = Theories)
            print(res)
            f.write(res)

