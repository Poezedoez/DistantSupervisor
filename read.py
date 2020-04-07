import json
import argparse
from collections import Counter, defaultdict
from evaluate import evaluate
import pandas as pd
import numpy as np
import sys
import csv

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