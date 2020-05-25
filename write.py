import json
import argparse
from collections import Counter
from evaluate import evaluate
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
from utils import create_dir
import shutil
import os
from read import read_ontology_entity_types


def print_dataset(input_path, output_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    file_object = open(output_path, 'w', encoding='utf-8') if output_path else sys.stdout
    # with open(output_path, 'w', encoding='utf-8') as f:
    for sentence in dataset:
        tokens = sentence["tokens"]
        print("[{}] \t {} \n".format(sentence["orig_id"], " ".join(tokens)), end='', file=file_object)
        for entity in sentence["entities"]:
            entity_tokens = tokens[entity["start"]:entity["end"]]
            print("[entity] \t {} \t {} \n".format(" ".join(entity_tokens), entity["type"]), end='', file=file_object)
        for relation in sentence["relations"]:
            entities = sentence["entities"]
            head_entity = entities[relation["head"]]
            tail_entity = entities[relation["tail"]]
            head_tokens = tokens[head_entity["start"]:head_entity["end"]]
            tail_tokens = tokens[tail_entity["start"]:tail_entity["end"]]
            print("[relation] \t {} \t |{}| \t {} \n".format(" ".join(head_tokens), relation["type"],
                                                            " ".join(tail_tokens)), end='', file=file_object)
        print('---------------------------------------------------------------------------------------------------------------------------------------------- \n', end='', file=file_object)
        print('\n', end='', file=file_object)


def print_sentences(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in dataset:
            f.write("{} \n".format(" ".join(sentence["tokens"])))


def print_statistics(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        stats = json.load(json_file)

    print("--- STATISTICS ---")
    print("Ontology version used: v{}".format(stats["ontology_version"]))
    print("Label function: {}".format([stats["label_function"]]))
    print("Threshold on embedding cosine similarity (cos_theta):", stats["cos_theta"])
    print("Processed {} sentences of which {} contained at least one entity".format(
        stats["sentences_processed"], stats["entity_sentences"]
    ))
    print("Time taken: {} seconds".format(stats["time_taken"]))
    print("--- Entities ---")
    tokens_per_entity = stats["tokens_total"] / stats["entities_total"] if(
        stats["entities_total"] != 0
    ) else 0
    print("Every {} tokens an entity occurs".format(tokens_per_entity))
    print("A total of {} token spans were labeled as entities".format(stats["entities_total"]))
    print("Entities were found in the following classes:")
    for type_, instance_counter in stats["entities"].items():
        count = sum([count for _, count in instance_counter.items()])
        print(type_, count)
    print("The most frequently labeled entities per class are:")
    for type_, instance_counter in stats["entities"].items():
        print("{} \t".format(type_), Counter(instance_counter).most_common(5))
    print("--- Relations ---")
    relations_per_sentence = stats["relation_candidates"]/stats["relations_total"] if(
        stats["relations_total"] != 0
    ) else 0
    print("Every {} sentences with at least two entities a relation occurs".format(relations_per_sentence))
    print("A total of {} entity pairs were labeled as relations".format(stats["relations_total"]))
    print("Relations were found in the following classes:")
    for relation, count in stats["relations"].items():
        print(relation, count)
    print()


def write_entities_without_duplicates(ontology_entities_path, dataset_path, output_path='entity_candidates.csv'):
    ontology_entities = read_ontology_entity_types(ontology_entities_path)
    entity_set = set([entity.lower() for entity in ontology_entities])
    with open(dataset_path, 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    with open(output_path, 'w') as csvfile:
        writer = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for sentence in dataset:
            entities = sentence["entities"]
            tokens = sentence["tokens"]
            for e in entities:
                entity_string = ' '.join(tokens[e["start"]:e["end"]]).lower()
                if entity_string in entity_set:
                    continue
                else:
                    writer.writerow([entity_string, e["type"], ' '.join(tokens)])
                    entity_set.add(entity_string)
    

def compare_datasets(gt_path, pred_path, output_path=None):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_dataset = json.load(f)

    assert len(gt_dataset)==len(pred_dataset)

    file_object = open(output_path, 'w', encoding='utf-8') if output_path else sys.stdout
    for gt_sentence, pred_sentence in zip(gt_dataset, pred_dataset):
        gt_tokens = gt_sentence["tokens"]
        print("|{}| {} \n".format(gt_sentence["orig_id"], " ".join(gt_tokens)), file=file_object)
        for entity in gt_sentence["entities"]:
            entity_tokens = gt_tokens[entity["start"]:entity["end"]]
            line = "[gold] \t {} \t {}".format(" ".join(entity_tokens), entity["type"])
            print(line, file=file_object)
        pred_tokens = pred_sentence["tokens"]
        for entity in pred_sentence["entities"]:
            entity_tokens = pred_tokens[entity["start"]:entity["end"]]
            line = "[pred] \t {} \t {}".format(" ".join(entity_tokens), entity["type"])
            print(line, file=file_object)
        print('-'*50, file=file_object)

def save_json(json_object, output_path):
    create_dir(output_path)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_object, json_file)


def save_list(l, output_path):
    create_dir(output_path)
    with open(output_path, 'w', encoding='utf-8') as txt_file:
        for item in l:
            txt_file.write("{} \n".format(item))


def save_copy(input_path, output_path):
    create_dir(output_path)
    if os.path.exists(input_path):
        shutil.copyfile(input_path, output_path)
    else:
        print("Cannot copy file: input path not found!")


def plot(cos_thetas, run_date="21_03_2020_11_51_28", set_="test", averaging="micro"):

    precisions = []
    recalls = []
    f1s = []

    for cos_theta in cos_thetas:
        gt_path = "data/{}/{}/{}/string_labeling/dataset.json".format(run_date, set_, cos_theta)
        pred_path =  "data/{}/{}/{}/embedding_labeling/dataset.json".format(run_date, set_, cos_theta)
        micro_macro_averages = evaluate(gt_path, pred_path)
        offset = 0 if averaging=="micro" else 3
        precisions.append(micro_macro_averages[0+offset])
        recalls.append(micro_macro_averages[1+offset])
        f1s.append(micro_macro_averages[2+offset])

    plt.title("Noun phrase embedding matches using different cosine similarity thresholds")
    plt.xticks(np.linspace(0.72, 1, 15))
    plt.plot(cos_thetas, precisions, label="precision")
    plt.plot(cos_thetas, recalls, label="recall")
    plt.plot(cos_thetas, f1s, label="f1")
    plt.xlabel(r'$cos( \theta )$')
    plt.ylabel("p/r/f1 ({})".format(averaging))

    plt.legend()
    plt.savefig("data/{}/plot_embedding_labeling_{}.png".format(run_date, averaging), dpi=720)

if __name__ == "__main__":
    in_path = 'data/DistantlySupervisedDatasets/26_03_2020_09_04_56/train/embedding_labeling/dataset.json'
    ontology_path = "data/ontology_entities.csv"
    write_entities_without_duplicates(ontology_path, in_path)

