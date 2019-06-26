import pandas as pd
import numpy as np
import json
import ast
from fuzzywuzzy import process
import Levenshtein
import argparse


def match_names(name,FBtaxonomy):
    best_candidates_distances=[]
    best_candidates_levels = []
    best_candidates_names = []
    for level in levels:
        best_candidates = process.extractBests(name, FBtaxonomy[level])
        for i in range(0,len(best_candidates)):
            candidate=best_candidates[i]
            if type(candidate[0])==str:
                distance = Levenshtein.ratio(name.lower(), candidate[0].lower())
                best_candidates_distances.insert(0,distance)
                best_candidates_levels.insert(0,level)
                best_candidates_names.insert(0,candidate[0])
    return best_candidates_distances,best_candidates_levels,best_candidates_names,\
           sorted(zip(best_candidates_distances,best_candidates_levels, best_candidates_names), reverse=True)[:1]


def look_fi_parents(taxonomy, name):
    _distances, _levels, _names, best_matches = match_names(name,taxonomy)
    all_parents=[]
    for i in range(0,len(best_matches)):
        level = best_matches[i][1]
        index_level = levels.index(level)
        id = taxonomy[taxonomy[levels[index_level]] == best_matches[i][2]].index[0]
        parents=[best_matches[i][2]]
        index_level = index_level-1
        while index_level>-1:
            index_parent=FBtaxonomy[levels[index_level]].iloc[:id].last_valid_index()
            parents.insert(0,FBtaxonomy[levels[index_level]].iloc[index_parent])
            index_level = index_level-1
        all_parents.append(parents)
    return best_matches,all_parents





def parse_args():
    parser = argparse.ArgumentParser(
        description="bootstrap")

    parser.add_argument("--taxonomy",
                        type=str,
                        required=True,
                        help="Path to taxonomy file")

    parser.add_argument("--test_result",
                        type=str,
                        required=True,
                        help="Path to the testing file")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    taxonomy = args.taxonomy
    test_result_path = args.test_result
    test_result = pd.read_csv(test_result_path, delimiter=';;', engine='python')
    FBtaxonomy = pd.read_csv(taxonomy,
                             names=['id', 'level_1', 'level_2', 'level_3', 'level_4', \
                                    'level_5', 'level_6'])
    levels = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5', 'level_6']
    new_items=0
    #for i in range(0,test_result.shape[0]-1):
    for i in range(0,test_result.shape[0]-1):
        sentence =  test_result.iloc[i,1]
        list_tuples = ast.literal_eval(sentence)
        fashion_items = [item for item in list_tuples if item[1] == 'PER']
        fashion_items_reduced=[]
        indexes_fashion_items = [list_tuples.index(fashion_items[i]) for i in range(0,len(fashion_items))]
        id=0
        while (id < len(indexes_fashion_items)):
            if (indexes_fashion_items[id]+1) in indexes_fashion_items:
                if (indexes_fashion_items[id] + 2) in indexes_fashion_items:
                    fashion_items_reduced.append(
                        list_tuples[indexes_fashion_items[id]][0] + ' ' + list_tuples[indexes_fashion_items[id] + 1][0] \
                        + ' ' + list_tuples[indexes_fashion_items[id] + 2][0])
                    id=id+3
                else:
                    fashion_items_reduced.append(list_tuples[indexes_fashion_items[id]][0]+' '+list_tuples[indexes_fashion_items[id]+1][0])
                    id = id + 2


            else:
                fashion_items_reduced.append(list_tuples[indexes_fashion_items[id]][0])
                id = id+1
        for i in range(0,len(fashion_items_reduced)):
            fashion_item = fashion_items_reduced[i]
            best_matches, all_parents = look_fi_parents (FBtaxonomy,fashion_item)
            if best_matches[0][0]>0.8:
                print('fashion item:',fashion_item,'\t best matches: \t',best_matches[0][2],'\t all parents:\t',all_parents)
            elif (len(fashion_item.split(' '))>1):
                list_items = fashion_item.split(' ')
                best_matches, all_parents = look_fi_parents(FBtaxonomy, list_items[-1])
                if best_matches[0][0] > 0.8:
                    new_items = new_items + 1
                    print('fashion item:', fashion_item, '\t suggested parent: \t', best_matches[0][2], '\t all parents:\t', all_parents)


    print("total number of discovered items:",new_items)