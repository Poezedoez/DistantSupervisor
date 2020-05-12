import json
import argparse
from collections import Counter, defaultdict
from evaluate import evaluate
import pandas as pd
import numpy as np
import sys
import csv
import os
import glob
import nltk
import re
from utils import glue_subtokens


def read_ontology_entities(path):
    ontology_entities = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip headers
        for _, class_, instance in csv_reader:
            ontology_entities[class_].append(instance)

    return ontology_entities


def read_ontology_relations(path):
    df = pd.read_csv(path)
    ontology_relations = {head: {} for head in df['head']}
    for head, tail, relation in zip(df['head'], df['tail'], df['relation']):
        ontology_relations[head][tail] = relation
            

    return ontology_relations


def read_ontology_types(ontology_path, relations_path):
    types = {}
    entities_df = pd.read_csv(ontology_path)
    relations_df = pd.read_csv(relations_path)
    types["entities"] = {type_:{"short": type_, "verbose": type_} for type_ in set(entities_df["Class"])}
    types["relations"] = {}
    for _, row in relations_df.iterrows():
        type_ = row["relation"]
        types["relations"][type_] = {"short": type_, "verbose": type_, "symmetric":row["symmetric"]}
    
    return types


def iter_sentences(data, selection=None, includes_special_tokens=True, filter_sentences=True):

    selected_documents = '{} to {}'.format(selection[0], selection[1]) if selection else 'all'
    print("Iterating over document range: {}".format(selected_documents))
    improper_sentences = 0
    for document_sentences, document_embeddings, doc_name in _read_documents(data, selection):
        extra = 1 if includes_special_tokens else 0
        offset = 0
        for sentence in document_sentences:
            subtokens_length = len(sentence) - (2 * extra)
            subtokens = sentence[extra:-extra]
            embeddings = document_embeddings[offset+extra:offset+extra+subtokens_length]
            if proper_sentence(subtokens) or not filter_sentences:
                yield subtokens, embeddings, doc_name
            else:
                glued_tokens, _, _ = glue_subtokens(subtokens)
                # print(glued_tokens)
                # print()
                improper_sentences += 1
            offset += len(sentence)
    print("improper sentences found", improper_sentences)


def proper_sentence(subtokens, symbol_threshold=0.2):
    ## Verb checking
    VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    glued_tokens, _, _ = glue_subtokens(subtokens)
    pos_tags = {t[1] for t in nltk.pos_tag(glued_tokens)}
    has_verbs = VERBS.intersection(pos_tags)

    ## Symbol checking
    alphabetical_sentence = False
    alphabet_tokens = len([s for s in glued_tokens if re.match("^[a-zA-Z]*$", s)])
    if float(alphabet_tokens)/float(len(glued_tokens)) > symbol_threshold:
        alphabetical_sentence = True
    proper = (has_verbs and alphabetical_sentence)
    # if not proper:
    #     v = "VERBS" if not has_verbs else ''
    #     a = "a-Z" if not alphabetical_sentence else ''
    #     print("The following sentence does not have {} {}".format(v, a))
    return (has_verbs and alphabetical_sentence)


def _read_documents(path, selection):
    flist = os.listdir(path) if not selection else os.listdir(path)[selection[0]:selection[1]]
    for folder in flist:
        text_path = glob.glob(path + "{}/representations/".format(folder) + "text_sentences|*.tokens")[0]
        with open(text_path, 'r', encoding='utf-8') as text_json:
            text = json.load(text_json)
        embeddings_path = glob.glob(path + "{}/representations/".format(folder) +
                                    "text_sentences|*word_embeddings.npy")[0]
        embeddings = np.load(embeddings_path)

        yield text, embeddings, folder

