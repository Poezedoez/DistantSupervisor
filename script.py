import os
import glob
import json
import nltk
from write import save_copy, save_json, print_evaluation_scores
from nltk.translate.ribes_score import position_of_ngram
from embedders import BertEmbedder
from Ontology import Ontology
from read import DataIterator
import numpy as np
import pandas as pd
from evaluate import evaluate as eval
from heuristics import RelationMatcher

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

def evaluate_ontology_representations():
    version = 4
    ontology_path = "data/ontology/v{}/".format(version)
    data_path = "data/ScientificDocuments/"
    results_path = "data/ontology/evaluation/v{}/".format(version)
    embedder = BertEmbedder('data/scibert_scivocab_cased')
    token_pooling = ["absmax", "max", "mean", "none", "absmax", "max", "mean", "none", "none", "none"]
    mention_pooling = ["none", "none", "none", "none", "absmax", "max", "mean", "absmax", "max", "mean"]
    
    # Init train iterator
    selection = (0, 500)
    train_iterator = DataIterator(
        data_path, 
        selection=selection, 
        includes_special_tokens=True, 
    )

    # Init eval iterator
    selection = (500, 700)
    eval_iterator = DataIterator(
        data_path, 
        selection=selection, 
        includes_special_tokens=True, 
    )

    results = {}
    for tp, mp in zip(token_pooling, mention_pooling):
        ontology = Ontology(ontology_path)
        ontology.calculate_entity_embeddings(train_iterator, embedder, tp, mp)
        similarity_scores = ontology.evaluate_entity_embeddings(eval_iterator, embedder, tp)
        results["T|{}|M|{}|".format(tp, mp)] = similarity_scores

    full_path = results_path+'evaluation_scores.json'
    save_json(results, full_path)
    print_evaluation_scores(full_path)


def context_consistency_scores(v=42, f_reduce="mean", filtered=True):
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
    selection = (500, 900)
    eval_iterator = DataIterator(
        data_path, 
        selection=selection, 
        includes_special_tokens=True, 
        filter_sentences=True
    )

    filter_option = "filtered" if filtered else "unfiltered"
    entity_embedding_path = "data/ontology/v{}_entity_embeddings_{}_{}.json".format(f_reduce, filter_option)
    ontology = Ontology(entities_path, relations_path, entity_embedding_path)
    if not ontology.entity_index:
        ontology.calculate_entity_embeddings(train_iterator, embedder, f_reduce)
        faiss_index.save(ontology.entity_index, ontology.entity_table, "entities_{}_{}".format(f_reduce, filter_option), 
            output_path+"faiss/")

    similarity_scores = ontology.evaluate_entity_embeddings(eval_iterator, embedder, f_reduce)
    stds = []
    entities = []
    for i, (entity, scores) in enumerate(similarity_scores.items()):
        if len(scores) > 1 and np.any(scores):
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


def merge_annotated_sets():
    parent_path = "data/annotation/annotated/"
    p1 = parent_path+"10_annotated_seed31.json"
    p2 = parent_path+"20_annotated_seed42.json"
    p3 = parent_path+"20_annotated_seed45.json"
    output_path = parent_path+"50_annotated_combined_31_42_45.json"

    d1 = json.load(open(p1))
    d2 = json.load(open(p2))
    d3 = json.load(open(p3))

    merged = d1+d2+d3

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f)

def test_overlap_evaluation():
    train = "data/DistantlySupervisedDatasets/23_03_2020_15_35_01/train/0.90/string_labeling/dataset.json"
    pred = "data/DistantlySupervisedDatasets/26_03_2020_09_04_56/dev/embedding_labeling/dataset.json"
    gt = "data/DistantlySupervisedDatasets/26_03_2020_09_04_56/dev/string_labeling/dataset.json"
    eval(gt, pred, train, split="all")

def test_relation_matcher():
    version = 4
    ontology_path = "data/ontology/v{}/".format(version)
    ontology = Ontology(ontology_path)
    rel_matcher = RelationMatcher(ontology)
    tokens = ["k", "-", "NN", "is", "a", "local", "classifier", "so", "k", "-", "means", "or", "any", "clustering", "algorithm", "can", "be", "used", "for", "NLP"]
    entities = [{"start":0, "end":3, "type":"mla"}, {"start":5, "end":7, "type":"mla"}, {"start":8, "end":11, "type":"mla"}, {"start":13, "end":15, "type":"mla"}, {"start":19, "end":20, "type":"aa"}]
    rel_matcher.pattern_match(tokens, entities, verbose=True)

def clean_ontology_v7():
    df = pd.read_csv('data/ontology/v7/ontology_entities_unfiltered.csv')
    unwanted_concepts = ["CONCEPT", "AI concept"]
    filtered = df[~(df.word_match =='no') & (~df.parent_name.isin(unwanted_concepts))]
    df_out = filtered.rename(columns={"final name":"Instance", "parent_name":"Class"})
    df_out[['Class', 'Instance']].to_csv('data/ontology/v7/ontology_entities.csv', index=False)


if __name__ == "__main__":
    # test_overlap_evaluation()
    # evaluate_ontology_representations()
    # test_relation_matcher()
    clean_ontology_v7()




