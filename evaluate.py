from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np
import json
import argparse
from typing import List, Tuple, Dict, Set
import read
from collections import defaultdict

# From spert.evaluator class
# https://github.com/markus-eberts/spert/blob/master/spert/evaluator.py

def _get_row(data, label):
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)

def _print_results(per_type: List, micro: List, macro: List, types: List):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)

def _compute_metrics(gt_all, pred_all, types, print_results: bool = False):
    labels = [t for t in types]
    per_type = prfs(gt_all, pred_all, labels=labels, average=None)
    micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

    return [m * 100 for m in micro + macro]

## Tuple = 
#   (start, end, entity_type, entity_phrase)
#   or
#   ((head_start, head_end, pseudo_head_type), (tail_start, tail_end, pseudo_tail_type), pred_rel_type, rel_phrase)
def _align_flat(gt: List[List[Tuple]], pred: List[List[Tuple]]):
    assert len(gt) == len(pred) # same amount of sequences, but not predictions per sequence

    gt_flat = []
    pred_flat = []
    phrases_flat = []
    types = set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()
        union.update(sample_gt)
        union.update(sample_pred)

        for s in union:
            phrases_flat.append(s[3])
            if s in sample_gt:
                t = s[2] if s[2].lower()!="other" else 0
                gt_flat.append(t)
                types.add(t)
            else:
                gt_flat.append("0")

            if s in sample_pred:
                t = s[2]
                pred_flat.append(t)
                types.add(t)
            else:
                pred_flat.append("0")
    
    return gt_flat, pred_flat, phrases_flat, types


def _score(gt_flat, pred_flat, overlap_flat, types, split="all", print_results=True):
    assert split.lower() in {"all", "exact", "partial", "new"}
    if split == "all":
        gt_split, pred_split = gt_flat, pred_flat
    else:
        gt_split, pred_split = [], []
        for gt, pred, overlap in zip(gt_flat, pred_flat, overlap_flat):
            if overlap.lower() == split:
                gt_split.append(gt)
                pred_split.append(pred)
    if not gt_split:
        print("No overlap occurrences of split type:", split)
        print("Evaluating on all split types")
        metrics = _compute_metrics(gt_flat, pred_flat, types, print_results)
    else:
        metrics = _compute_metrics(gt_split, pred_split, types, print_results)

    return metrics


def _convert_entities(sequence):
    entity_tuples = []
    entities = sequence["entities"]
    tokens = sequence["tokens"]
    for entity in entities:
        phrase = " ".join(tokens[entity["start"]:entity["end"]])
        tuple_ = (entity["start"], entity["end"], entity["type"], phrase)
        entity_tuples.append(tuple_)
        
    return entity_tuples


def _convert_relations(sequence, pseudo_type="ENTITY"):
    relation_tuples = []
    entities = sequence["entities"]
    relations = sequence["relations"]
    tokens = sequence["tokens"]
    for relation in relations:
        # head
        head_entity = entities[relation["head"]]
        head_phrase = "_".join(tokens[head_entity["start"]:head_entity["end"]])
        head_tuple = (head_entity["start"], head_entity["end"], pseudo_type)

        # tail
        tail_entity = entities[relation["tail"]]
        tail_phrase = "_".join(tokens[tail_entity["start"]:tail_entity["end"]])
        tail_tuple = (tail_entity["start"], tail_entity["end"], pseudo_type)

        # combined
        relation_phrase = " ".join([head_phrase, tail_phrase])
        relation_tuple = (head_tuple, tail_tuple, relation["type"], relation_phrase)
        relation_tuples.append(relation_tuple)
        
    return relation_tuples


def _convert_train(train_dataset, f_conversion):
    stop_words = set(". , : ; ? ! [ ] ( ) \" ' % - the on 's of a an".split())
    train_exact = defaultdict(lambda: set())
    train_partial = defaultdict(lambda: set())
    for sequence in train_dataset:
        conversions = f_conversion(sequence)
        for c in conversions:
            type_, phrase = c[2], c[-1]
            train_exact[type_].add(phrase)
            partial_terms = set(phrase.split())-stop_words
            train_partial[type_].update(partial_terms)

    print(train_exact)
    return train_exact, train_partial

def _measure_overlap(phrases, gt_types, train_exact, train_partial):
    assert(len(phrases)==len(gt_types))
    overlaps = []
    for phrase, type_ in zip(phrases, gt_types):
        if phrase in train_exact[type_]:
           overlap = "exact"
        elif set(phrase.split()).intersection(train_partial[type_]): 
            overlap = "partial"
        else:
            overlap = "new"
        overlaps.append(overlap)
    
    return overlaps


def evaluate(gt_path, pred_path, train_path=None, split="all"):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_dataset = json.load(f)

    if train_path:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
    else:
        train_dataset = []

    train_exact_entities, train_partial_entities = _convert_train(train_dataset, _convert_entities)
    train_exact_relations, train_partial_relations = _convert_train(train_dataset, _convert_relations)

    gt_entities = []
    pred_entities = []

    gt_relations = []
    pred_relations = []

    for gt_sequence, pred_sequence in zip(gt_dataset, pred_dataset):
        gt_entity_tuples = _convert_entities(gt_sequence)
        pred_entity_tuples = _convert_entities(pred_sequence)
        gt_entities.append(gt_entity_tuples)
        pred_entities.append(pred_entity_tuples)

        gt_relation_tuples = _convert_relations(gt_sequence)
        pred_relation_tuples = _convert_relations(pred_sequence)
        gt_relations.append(gt_relation_tuples)
        pred_relations.append(pred_relation_tuples)

    gt_entities, pred_entities, entity_phrases, entity_types = _align_flat(gt_entities, pred_entities)
    # print(gt_entities, pred_entities, entity_phrases, entity_types)
    gt_relations, pred_relations, relation_phrases, relation_types = _align_flat(gt_relations, pred_relations)
    entity_overlap = _measure_overlap(entity_phrases, gt_entities, 
                                      train_exact_entities, train_partial_entities)
    relation_overlap = _measure_overlap(relation_phrases, gt_relations, 
                                        train_exact_relations, train_partial_relations)

    print("")
    print("--- Entities (named entity recognition (NER)) ---")
    print("An entity is considered correct if the entity type and span is predicted correctly")
    print("")
    ner_eval = _score(gt_entities, pred_entities, entity_overlap, entity_types, split, print_results=True)
    print("")
    print("--- Relations ---")
    print("")
    print("A relation is considered correct if the relation type and the two "
            "related entity spans are predicted correctly")
    print("")
    rel_eval = _score(gt_relations, pred_relations, relation_overlap, relation_types, split, print_results=True)

    return ner_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate spert json formatted dataset')
    parser.add_argument('gt_path', type=str, help='path to the ground truth dataset.json file')
    parser.add_argument('pred_path', type=str, help='path to the predicted dataset.json file')
    parser.add_argument('--train_path', type=str, help='path to the train dataset.json file', default=None)
    parser.add_argument('--split', type=str, help='name of the overlap to do a split evaluation', default="all")

    args = parser.parse_args()
    evaluate(args.gt_path, args.pred_path, args.train_path, args.split)