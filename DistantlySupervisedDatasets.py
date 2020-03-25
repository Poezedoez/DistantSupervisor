import json
from embedders import BertEmbedder
import os
import glob
import numpy as np
from pathlib import Path
import time
from utils import KnuthMorrisPratt, merge_list_dicts
from collections import defaultdict, Counter
import argparse
import csv
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from outputs import print_dataset, print_statistics, read_types
import pandas as pd
import copy


def _read_ontology_entities(path):
    ontology_entities = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip headers
        for _, class_, instance in csv_reader:
            ontology_entities[class_].append(instance)

    return ontology_entities


def _read_ontology_relations(path):
    df = pd.read_csv(path)
    ontology_relations = {head: {} for head in df['head']}
    for head, tail, relation in zip(df['head'], df['tail'], df['relation']):
        ontology_relations[head][tail] = relation
            

    return ontology_relations


def _glue_subtokens(subtokens):
    glued_tokens = []
    tok2glued = []
    glued2tok = []
    for i, token in enumerate(subtokens):
        if token.startswith('##'):
            glued_tokens[len(glued_tokens) - 1] = glued_tokens[len(glued_tokens) - 1] + token.replace('##', '')
        else:
            glued2tok.append(i)
            glued_tokens.append(token)

        tok2glued.append(len(glued_tokens) - 1)

    return glued_tokens, tok2glued, glued2tok


class DistantlySupervisedDatasets:
    """
    Args:
        ontology_path (str): path to the ontology in json format
        document_path (str): path to the parent folder of scientific documents containing
            subfolders named with document id's
        entity_embedding_path (str): path to the precalculated entity embeddings of the ontology
        output_path (str): path to store results
    Attr:
        ontology (dict): ontology loaded with json from the given path
        embedder (Embedder): embedder used for tokenization and obtaining embeddings
        timestamp (str): string containing object creation time
        document_path (str): stored document path from init argument
        entity_embedding_path (str): stored entity embedding path from init argument
        output_path (str): stored output path from init argument
        statistics (dict): dict to store statistics from creation process
        type_arrays (dict): dict of instance embeddings stacked for one class
        flist (list): list of files used
        dataset (list): list of annotated sentence datapoints
        
    """

    def __init__(
            self,
            ontology_entities_path="data/ontology_entities.csv",
            ontology_relations_path="data/ontology_relations.csv",
            document_path="data/ScientificDocuments/",
            entity_embedding_path="data/entity_embeddings.json",
            output_path="data/DistantlySupervisedDatasets/",
            timestamp=time.strftime("%Y%m%d-%H%M%S"),
            cos_theta=0.85
    ):

        self.ontology_entities = _read_ontology_entities(ontology_entities_path)
        self.ontology_relations = _read_ontology_relations(ontology_relations_path)
        self.types = read_types(ontology_entities_path, ontology_relations_path)
        self.embedder = BertEmbedder('data/scibert_scivocab_cased')
        self.timestamp = timestamp
        self.document_path = document_path
        self.entity_embedding_path = entity_embedding_path
        self.output_path = output_path
        self.cos_theta = cos_theta
        self.type_arrays = {}
        self.index_to_string = {}
        self.flist = []
        self.label_function_names = {0: "string_labeling", 1: "embedding_labeling", 2: "combined_labeling"}
        # self.label_functions = {0: self._string_match, 1: self._embedding_match, 2: self._combined_match}
        self.datasets = {"string_labeling": [], "embedding_labeling": [], "combined_labeling": []}
        self.label_statistics, self.global_statistics = self._prepare_statistics()

    def create(self, label_function=0, selection=None):
        if label_function > 0:
            self._load_type_arrays()
        start_time = time.time()
        for sentence_subtokens, sentence_embeddings in self._iter_sentences(selection):
            self._label_sentence(sentence_subtokens, sentence_embeddings, label_function)
        end_time = time.time()
        self.global_statistics["time_taken"] = int(end_time - start_time)
        self._save()

    def _save(self):       
        # Save datasets of different labeling functions
        for label_function, dataset in self.datasets.items():
            if not dataset:
                continue
            dataset_path = self.output_path + '{}/dataset.json'.format(label_function)
            directory = os.path.dirname(dataset_path)
            Path(directory).mkdir(parents=True, exist_ok=True)
            with open(dataset_path, 'w', encoding='utf-8') as json_file:
                json.dump(dataset, json_file)

            # Save dataset statistics
            statistics_path = self.output_path + '{}/statistics.json'.format(label_function)
            self.global_statistics["label_function"] = label_function
            self.label_statistics[label_function].update(self.global_statistics)

            with open(statistics_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.label_statistics[label_function], json_file)
            print_statistics(statistics_path)
            
            # Save pretty output of labeled examples
            print_dataset(dataset_path, self.output_path+'{}/classified_examples.txt'.format(label_function))

        # Save ontology used
        shutil.copyfile(args.ontology_entities_path, self.output_path + 'ontology_entities.csv')
        shutil.copyfile(args.ontology_relations_path, self.output_path + 'ontology_relations.csv')

        # Save ontology types
        with open(self.output_path+'ontology_types.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.types, json_file)

        # Save list of selected documents used for the split
        with open(self.output_path + 'filelist.txt', 'w', encoding='utf-8') as txt_file:
            for file in self.flist:
                txt_file.write("{} \n".format(file))

    def _iter_sentences(self, selection=None, includes_special_tokens=True):
        for document_sentences, document_embeddings in self._read_documents(selection):
            extra = 1 if includes_special_tokens else 0
            offset = 0
            for sentence in document_sentences:
                subtokens_length = len(sentence) - (2 * extra)
                yield sentence[extra:-extra], document_embeddings[offset+extra:offset+extra+subtokens_length]
                offset += len(sentence)

    def _read_documents(self, selection=None):
        path = self.document_path
        self.flist = os.listdir(path) if not selection else os.listdir(path)[selection[0]:selection[1]]
        for folder in self.flist:
            text_path = glob.glob(path + "{}/representations/".format(folder) + "text_sentences|*.tokens")[0]
            with open(text_path, 'r', encoding='utf-8') as text_json:
                text = json.load(text_json)
            embeddings_path = glob.glob(path + "{}/representations/".format(folder) +
                                        "text_sentences|*word_embeddings.npy")[0]
            embeddings = np.load(embeddings_path)

            yield text, embeddings
    
    def _prepare_statistics(self):
        label_statistics = {"relations": Counter(), "relations_total": 0,
                    "entities": {type_: Counter() for type_ in self.ontology_entities},
                    "entity_sentences": 0, "entities_total": 0, "tokens_total": 0,
                    "relation_candidates": 0}
        global_statistics = {}
        global_statistics["sentences_processed"] = 0
        global_statistics["cos_theta"] = self.cos_theta

        return {dataset: copy.deepcopy(label_statistics) for dataset in self.datasets}, global_statistics

    def _string_match(self, tokens, execute=True):
        matches = {type_: [] for type_ in self.ontology_entities}
        if not execute:
            return matches
        tokens = [token.lower() for token in tokens]
        for type_, string_instances in self.ontology_entities.items():
            for string_instance in string_instances:
                tokenized_string = [token.lower() for token in self.embedder.tokenize(string_instance)]
                glued_string, _, _ = _glue_subtokens(tokenized_string)
                string_length = len(glued_string)
                matches[type_] += [(occ, occ + string_length) for occ in KnuthMorrisPratt(tokens, glued_string)]

        return matches

    def _embedding_match(self, sentence_embeddings, glued2tok, glued_tokens, execute=True, threshold=0.80):
        matches = {type_: [] for type_ in self.ontology_entities}
        if not execute:
            return matches
        for type_ in self.type_arrays:
            prev_entity = False
            start = 0
            similarities = cosine_similarity(sentence_embeddings, self.type_arrays[type_])
            max_similarities = similarities.max(axis=1)
            # max_indices = similarities.argmax(axis=1)
            score = 0
            for i, token_pointer in enumerate(glued2tok):
                score = max_similarities[token_pointer]
                # if score > threshold:
                #     print(score, glued_tokens[i], self.index_to_string[type_][max_indices[token_pointer]])
                # entity span starts
                if score > threshold and not prev_entity:
                    start = i
                    prev_entity = True
                    continue

                # entity span continues
                elif score > threshold and prev_entity:
                    prev_entity = True
                    continue

                # entity span ends
                elif prev_entity:
                    matches[type_].append((start, i))

                start = i
                prev_entity = False

            # last token of the sentence is entity
            if score > threshold:
                matches[type_].append((start, token_pointer+1))

        return matches

    def _combined_match(self, string_matches, embedding_matches, execute=True):
        matches = {type_: [] for type_ in self.ontology_entities}
        if not execute:
            return matches
        matches = merge_list_dicts(string_matches, embedding_matches)

        return matches

    def _label_sentence(self, sentence_subtokens, sentence_embeddings, label_function=0):
        def label_relations(entities):
            relations = []
            if len(entities) < 2:
                return relations
            pairs = [(a, b) for a in range(0, len(entities)) for b in range(0, len(entities))]
            for head_index, tail_index in pairs:
                head = entities[head_index]["type"]
                tail = entities[tail_index]["type"]
                relation = self.ontology_relations.get(head, {}).get(tail, None)
                if relation:
                    relations.append({"type": relation, "head": head_index, "tail": tail_index})

            return relations

        def label_entities(matches):
            entities = []
            for type_, positions in matches.items():
                for position in positions:
                    start, end = position
                    entities.append({"type": type_, "start": start, "end": end})

            return entities

        glued_tokens, _, glued2tok = _glue_subtokens(sentence_subtokens)

        # Find string entity matches
        do_string_matching = (label_function == 0 or label_function == 2)
        string_matches = self._string_match(glued_tokens, do_string_matching)
        string_entities = label_entities(string_matches)
        string_relations = label_relations(string_entities)

        # Find embedding entity matches
        do_embedding_matching = (label_function == 1 or label_function == 2)
        embedding_matches = self._embedding_match(sentence_embeddings, glued2tok, glued_tokens,
            do_embedding_matching, threshold=self.cos_theta)
        embedding_entities = label_entities(embedding_matches)
        embedding_relations = label_relations(embedding_entities)

        # Find combined entity matches
        do_combined_matching = (label_function == 2)
        combined_matches = self._combined_match(string_matches, embedding_matches, do_combined_matching)
        combined_entities = label_entities(combined_matches)
        combined_relations = label_relations(combined_entities)
        
        self.global_statistics["sentences_processed"] += 1
        if not string_entities: # use all sentences with at least one string match
            return

        self._add_training_instance(glued_tokens, string_entities, string_relations, "string_labeling")
        self._add_training_instance(glued_tokens, embedding_entities, embedding_relations, "embedding_labeling")
        self._add_training_instance(glued_tokens, combined_entities, combined_relations, "combined_labeling")


    def _add_training_instance(self, tokens, entities, relations, label_function):
        self._log_statistics(tokens, entities, relations, label_function)
        joint_string = "".join(tokens)
        hash_string = hash(joint_string)
        training_instance = {"tokens": tokens, "entities": entities,
                             "relations": relations, "orig_id": hash_string}
        self.datasets[label_function].append(training_instance)

    def _log_statistics(self, tokens, entities, relations, label_function):
        # Log entity statistics
        self.label_statistics[label_function]["tokens_total"] += len(tokens)
        self.label_statistics[label_function]["entities_total"] += len(entities)
        if entities:
            self.label_statistics[label_function]["entity_sentences"] += 1
            for entity in entities:
                start, end = entity["start"], entity["end"]
                type_ = entity["type"]
                entity_string = "_".join(tokens[start:end]).lower()
                self.label_statistics[label_function]["entities"][type_][entity_string] += 1
                # print("Found |{}| as |{}| using |{}|".format(entity_string, type_, label_function))

        # Log relation statistics
        self.label_statistics[label_function]["relations_total"] += len(relations)
        if len(entities) > 1:
            self.label_statistics[label_function]["relation_candidates"] += 1
        if relations:
            for relation in relations:
                self.label_statistics[label_function]["relations"][relation["type"]] += 1

    def _load_type_arrays(self):
        def _calculate_entity_embeddings():
            # Sum all entity instances
            entity_embeddings = {type_: defaultdict(lambda: np.zeros(768)) for type_ in self.ontology_entities}
            entity_counter = {type_: Counter() for type_ in self.ontology_entities.keys()}
            for sentence_subtokens, sentence_embeddings in self._iter_sentences(selection=args.selection):
                glued_tokens, _, glued2tok = _glue_subtokens(sentence_subtokens)
                string_matches = self._string_match(glued_tokens)
                for type_, positions in string_matches.items():
                    for position in positions:
                        start, end = position
                        pointers = glued2tok[start:end+1]
                        if len(pointers) == 1:  # last token in sentence
                            pointers.append(pointers[-1]+1)
                        matched_embeddings = sentence_embeddings[pointers[0]:pointers[-1]]
                        matched_glued_tokens = glued_tokens[start:end]
                        embedding = np.stack(matched_embeddings).mean(axis=0)
                        entity_embeddings[type_][" ".join(matched_glued_tokens)] += embedding
                        entity_counter[type_][" ".join(matched_glued_tokens)] += 1

            # Average the sum of embeddings
            for type_, count_dict in entity_counter.items():
                for token, count in count_dict.items():
                    summed_embedding = entity_embeddings[type_][token]
                    entity_embeddings[type_][token] = (summed_embedding / count).tolist()

            return entity_embeddings

        if os.path.isfile(self.entity_embedding_path):
            with open(self.entity_embedding_path, 'r', encoding='utf-8') as json_file:
                entity_embeddings = json.load(json_file)
        else:
            entity_embeddings = _calculate_entity_embeddings()
            with open(self.entity_embedding_path, 'w', encoding='utf-8') as json_file:
                json.dump(entity_embeddings, json_file)

        index_to_string = {type_: {} for type_ in entity_embeddings}
        for type_ in entity_embeddings:
            embeddings = []
            for instance in entity_embeddings[type_]:
                index_to_string[type_][len(embeddings)] = instance
                embeddings.append(entity_embeddings[type_][instance])
            embeddings = [np.zeros(768)] if not embeddings else embeddings
            type_array = np.stack(embeddings)
            self.type_arrays[type_] = type_array

        self.index_to_string = index_to_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a distantly supervised dataset of scientific documents')
    parser.add_argument('--ontology_entities_path', type=str, default="data/ontology_entities.csv",
                        help="path to the ontology entities file")
    parser.add_argument('--ontology_relations_path', type=str, default="data/ontology_relations.csv",
                        help="path to the ontology relations file")
    parser.add_argument('--document_path', type=str, help='path to the folder containing scientific documents',
                        default="data/ScientificDocuments/")
    parser.add_argument('--output_path', type=str, default="data/DistantlySupervisedDatasets/", help="output path")
    parser.add_argument('--entity_embedding_path', type=str, default="data/entity_embeddings.json",
                        help="path to file of precalculated lexical embeddings of the entities")
    parser.add_argument('--selection', type=int, nargs=2, default=None,
                        help="start and end of file range for train/test split")
    parser.add_argument('--label_function', type=int, default=2, choices=range(0, 3),
                        help="0 = string, 1 = embedding, 2 = string + embedding")
    parser.add_argument('--timestamp', type=str, default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--cos_theta', type=float, default=0.85,
                        help="similarity threshold for embedding based labeling")
    args = parser.parse_args()
    dataset = DistantlySupervisedDatasets(args.ontology_entities_path, args.ontology_relations_path, args.document_path,
                                         args.entity_embedding_path, args.output_path, args.timestamp, args.cos_theta)
    dataset.create(label_function=args.label_function, selection=tuple(args.selection))
