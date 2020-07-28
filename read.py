import json
import argparse
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import sys
import csv
import os
import glob
import nltk
import re
from heuristics import proper_sequence, RelationPattern
from embedders import glue_subtokens
import faiss
import math


class DataIterator:
    def __init__(self, data, selection=None, includes_special_tokens=True, filter_sentences=True):
        self.data = data
        self.selection = selection
        self.includes_special_tokens = includes_special_tokens
        self.filter_sentences = filter_sentences
    
    def iter_sentences(self):
        selected_documents = '{} to {}'.format(
            self.selection[0], self.selection[1]) if self.selection else 'all'
        print("Iterating over document range: {}".format(selected_documents))
        improper_sentences = 0
        for document_sentences, document_embeddings, doc_name in self._read_documents():
            extra = 1 if self.includes_special_tokens else 0
            offset = 0
            for sentence in document_sentences:
                subtokens_length = len(sentence) - (2 * extra)
                subtokens = sentence[extra:-extra]
                embeddings = document_embeddings[offset+extra:offset+extra+subtokens_length]
                glued_tokens, _, _ = glue_subtokens(subtokens)
                if proper_sequence(glued_tokens) or not self.filter_sentences:
                    yield subtokens, embeddings, doc_name
                else:
                    improper_sentences += 1
                offset += len(sentence)
        print("improper sentences found", improper_sentences)

    def _read_documents(self):
        flist = os.listdir(self.data) if not self.selection else os.listdir(
            self.data)[self.selection[0]:self.selection[1]]
        
        for folder in flist:
            text_path = glob.glob(self.data + "{}/representations/".format(folder) + "text_sentences|*.tokens")[0]
            with open(text_path, 'r', encoding='utf-8') as text_json:
                text = json.load(text_json)
            embeddings_path = glob.glob(self.data + "{}/representations/".format(folder) +
                                        "text_sentences|*word_embeddings.npy")[0]
            embeddings = np.load(embeddings_path)

            yield text, embeddings, folder


def read_ontology_entity_types(path, fraction=1.0, verbose=True):
    df = pd.read_csv(path)
    df["Class"] = df["Class"].str.lower()
    new_df = pd.DataFrame(columns=df.columns)
    unique_classes = set(df["Class"].tolist())
    for c in unique_classes:
        filtered = df[df["Class"]==c]
        class_count = len(filtered)
        samples = math.ceil(fraction*class_count)
        if verbose:
            print("sampled {} out of {} for |{}|:".format(samples, class_count, c))
        selection = filtered.sample(n=samples)
        new_df = new_df.append(selection, ignore_index=True)
    
    ontology_entities = dict(zip(new_df["Instance"], df["Class"]))

    return ontology_entities


def read_ontology_relation_types(path):
    df = pd.read_csv(path)
    ontology_relations = {head: {} for head in df['head']}
    for head, tail, relation in zip(df['head'], df['tail'], df['relation']):
        ontology_relations[head][tail] = relation
            
    return ontology_relations


def read_relation_patterns(path):
    df = pd.read_csv(path)
    relation_patterns = []
    for index, row in df.iterrows():
        relation_patterns.append(RelationPattern(row["regex"], row["relation_type"], 
                                         row["subject_position"], row["subject"]))

    return relation_patterns

if __name__ == "__main__":
    read_ontology_entity_types('data/ontology/v7/ontology_entities.csv', fraction=0.1)