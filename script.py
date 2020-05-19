import os
import glob
import json
import nltk
from write import save_copy, save_json
from nltk.translate.ribes_score import position_of_ngram
from embedders import BertEmbedder
from Ontology import Ontology
from read import DataIterator
import numpy as np

def read_full_texts(path):
    flist = os.listdir(path)
    with open("full_texts.txt", 'w', encoding='utf-8') as out_file:
        for folder in flist:
            text_path = glob.glob(path + "{}/representations/".format(folder) + "text|*")[0]
            with open(text_path, 'r', encoding='utf-8') as text_json:
                text = json.load(text_json)["value"]
                out_file.write(text)
                out_file.write('\n')

def noun_phrases(tokens):
    grammar = r"""
    NALL: {<NN>*<NNS>*<NNP>*<NNPS>*}
    NP: {<JJ>*<NALL>+}  

    """

    cp = nltk.RegexpParser(grammar)
    result = cp.parse(nltk.pos_tag(tokens))
    noun_phrases = []
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        np = ''
        for x in subtree.leaves():
            np = np + ' ' + x[0]
        noun_phrases.append(np.strip())

    spans = []
    for np in noun_phrases:
        splitted_np = np.split()
        start = position_of_ngram(tuple(splitted_np), tokens)
        end = start+len(splitted_np)
        spans.append((start, end))

    return noun_phrases, spans

def evaluate_ontology_representations(v=4):
    data_path = "data/ScientificDocuments/"
    entities_path = "data/ontology/v{}_ontology_entities.csv".format(v)
    relations_path = "data/ontology/v{}_ontology_relations.csv".format(v)
    results_path = "data/ontology/evaluation/"
    embedder = BertEmbedder('data/scibert_scivocab_cased')
    strategies = ["absmax_filtered", "absmax_unfiltered", "max_filtered", "max_unfiltered", "mean_filtered", "mean_unfiltered"]
    reductions = ["abs_max", "abs_max", "max", "max", "mean", "mean"]
    filters = [True, False, True, False, True, False]
    results = {}
    for s, r, f in zip(strategies, reductions, filters):

        # Init train iterator
        selection = (0, 500)
        train_iterator = DataIterator(
            data_path, 
            selection=selection, 
            includes_special_tokens=True, 
            filter_sentences=f
        )

        # Init eval iterator
        selection = (500, 700)
        eval_iterator = DataIterator(
            data_path, 
            selection=selection, 
            includes_special_tokens=True, 
            filter_sentences=f
        )

        save_path = "data/ontology/v{}_entity_embeddings_{}.json".format(v, s)
        ontology = Ontology(entities_path, relations_path)
        ontology.calculate_entity_embeddings(train_iterator, embedder, r)
        similarity_scores = ontology.evaluate_entity_embeddings(eval_iterator, embedder, r)
        save_json(ontology.entity_embeddings, save_path)
        results[s] = similarity_scores

    save_json(results, results_path+'evaluation_scores.json')

def context_consistency_scores(v=42, f_reduce="mean"):
    data_path = "data/ScientificDocuments/"
    entities_path = "data/ontology/v{}_ontology_entities.csv".format(v)
    relations_path = "data/ontology/v{}_ontology_relations.csv".format(v)
    results_path = "data/ontology/evaluation/"
    embedder = BertEmbedder('data/scibert_scivocab_cased')
    context_consistency_scores = {}

    # Init train iterator
    selection = (0, 500)
    train_iterator = DataIterator(
        data_path, 
        selection=selection, 
        includes_special_tokens=True, 
        filter_sentences=True
    )

    # Init eval iterator
    selection = (500, 700)
    eval_iterator = DataIterator(
        data_path, 
        selection=selection, 
        includes_special_tokens=True, 
        filter_sentences=True
    )

    save_path = "data/ontology/v{}_entity_embeddings_{}_filtered.json".format(v, f_reduce)
    ontology = Ontology(entities_path, relations_path)
    ontology.calculate_entity_embeddings(train_iterator, embedder, f_reduce)
    save_json(ontology.entity_embeddings, save_path)
    similarity_scores = ontology.evaluate_entity_embeddings(eval_iterator, embedder, f_reduce)
    stds = []
    entities = []
    for i, (entity, scores) in enumerate(similarity_scores.items()):
        if np.any(scores):
            std = np.std(scores)
            stds.append(std)
            entities.append(entity)

    a = stds-np.min(stds)
    b = np.max(stds)-np.min(stds)
    normalized_stds = np.divide(a, b, out=np.zeros_like(a), where=b!=0)    
    for entity, cis in zip(entities, normalized_stds):
        ccs = 1-cis
        context_consistency_scores[entity] = ccs
        print(entity, ccs)

    save_json(context_consistency_scores, results_path+'v{}_context_consistency_scores.json'.format(v))

if __name__ == "__main__":
    # evaluate_ontology_representations()
    context_consistency_scores()



