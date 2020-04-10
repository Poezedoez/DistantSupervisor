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
from outputs import print_dataset, print_statistics
from read import read_ontology_entities, read_ontology_relations, read_ontology_types
from utils import glue_subtokens, create_dir
import pandas as pd
import copy


class DistantlySupervisedDatasets:
    """
    Args:
        ontology_entities_path (str): path to the ontology entities csv file
        ontology_relations_path (str): path to the ontology relations csv file
        data_path (str): path to the folder of scientific documents containing
            named with document id's
        entity_embedding_path (str): path to the precalculated entity embeddings of the ontology
        output_path (str): path to store results
        timestamp_given (bool): whether a time stamp is included in the output path
        cos_theta (float): similarity threshold for embedding labeling

    Attr:
        ontology (dict): ontology loaded with json from the given path
        embedder (Embedder): embedder used for tokenization and obtaining embeddings
        timestamp (str): string containing object creation time
        data_path (str): stored document path from init argument
        entity_embedding_path (str): stored entity embedding path from init argument
        output_path (str): stored output path from init argument
        label_statistics (dict): dict to store statistics per label function
        global_statistics (dict): dict to store shared statistics between different label functions
        type_arrays (dict): dict of instance embeddings stacked for one entity type
        flist (list): list of files used
        label_function_names (dict): mapping of label function (int) to its name (str)
        datasets (dict): list of annotated sentence datapoints
        label_statistics (dict): statistics per labeling function
        global_statistics (list): globally shared statistics
        
    """

    def __init__(
            self,
            ontology_entities_path="data/ontology/ontology_entities.csv",
            ontology_relations_path="data/ontology/ontology_relations.csv",
            data_path="data/ScientificDocuments/",
            entity_embedding_path="data/ontology/entity_embeddings.json",
            output_path="data/DistantlySupervisedDatasets/",
            timestamp_given=False,
            cos_theta=0.83
    ):

        self.ontology_entities_path = ontology_entities_path
        self.ontology_relations_path = ontology_relations_path
        self.ontology_entities = read_ontology_entities(ontology_entities_path)
        self.ontology_relations = read_ontology_relations(ontology_relations_path)
        self.types = read_ontology_types(ontology_entities_path, ontology_relations_path)
        self.embedder = BertEmbedder('data/scibert_scivocab_cased')
        self.timestamp = '' if timestamp_given else time.strftime("%Y%m%d-%H%M%S")+'/'
        self.data_path = data_path
        self.entity_embedding_path = entity_embedding_path
        self.entity_embeddings = None
        self.output_path = output_path + self.timestamp
        self.cos_theta = cos_theta
        self.type_arrays = {}
        self.flist = []
        self.label_function_names = {0: "string_labeling", 1: "embedding_labeling", 2: "combined_labeling"}
        self.datasets = {"string_labeling": [], "embedding_labeling": [], "combined_labeling": []}
        self.label_statistics, self.global_statistics = self._prepare_statistics()

    def create(self, label_function=0, selection=None):
        # print("Number of processors available to use:", len(os.sched_getaffinity(0)))
        if label_function > 0:
            self._load_type_arrays(selection)
        start_time = time.time()
        print("Creating dataset...")
        for sentence_subtokens, sentence_embeddings in self._iter_sentences(selection):
            self._label_sentence(sentence_subtokens, sentence_embeddings, label_function)
        end_time = time.time()
        self.global_statistics["time_taken"] = int(end_time - start_time)
        self._save()

    def _save(self):       
        # Save datasets of different labeling functions
        for label_function, dataset in self.datasets.items():
            if self.label_statistics[label_function]["skip"]: # skip empty dataset
                continue
            dataset_path = self.output_path + '{}/dataset.json'.format(label_function)
            create_dir(dataset_path)
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
        shutil.copyfile(self.ontology_entities_path, self.output_path + 'ontology_entities.csv')
        shutil.copyfile(self.ontology_relations_path, self.output_path + 'ontology_relations.csv')

        # Save used lexical ontology embeddings
        output_path = self.output_path + 'entity_embeddings.json'
        create_dir(output_path)
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.entity_embeddings, json_file)
        if os.path.exists(self.entity_embedding_path):
            shutil.copyfile(self.entity_embedding_path, self.output_path + 'entity_embeddings.json')

        # Save ontology types
        with open(self.output_path+'ontology_types.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.types, json_file)

        # Save list of selected documents used for the split
        with open(self.output_path + 'filelist.txt', 'w', encoding='utf-8') as txt_file:
            for file in self.flist:
                txt_file.write("{} \n".format(file))

    def _iter_sentences(self, selection=None, includes_special_tokens=True):
        selected_documents = '{} to {}'.format(selection[0], selection[1]) if selection else 'all'
        print("Iterating over document range: {}".format(selected_documents))
        for document_sentences, document_embeddings in self._read_documents(selection):
            extra = 1 if includes_special_tokens else 0
            offset = 0
            for sentence in document_sentences:
                subtokens_length = len(sentence) - (2 * extra)
                yield sentence[extra:-extra], document_embeddings[offset+extra:offset+extra+subtokens_length]
                offset += len(sentence)

    def _read_documents(self, selection=None):
        path = self.data_path
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
                    "relation_candidates": 0, "skip":True}
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
                glued_string, _, _ = glue_subtokens(tokenized_string)
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
            score = 0
            for i, token_pointer in enumerate(glued2tok):
                score = max_similarities[token_pointer]
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
                matches[type_].append((start, i+1))

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

        glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)

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
        if not string_entities and label_function != 1: # use all sentences with at least one string match
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
            self.label_statistics[label_function]["skip"] = False
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

    def _load_type_arrays(self, selection):
        def _calculate_entity_embeddings(selection):
            # Sum all entity instances
            print("Calculating ontology entity embeddings...")
            entity_embeddings = {type_: defaultdict(lambda: np.zeros(768)) for type_ in self.ontology_entities}
            entity_counter = {type_: Counter() for type_ in self.ontology_entities.keys()}
            for sentence_subtokens, sentence_embeddings in self._iter_sentences(selection=selection):
                glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)
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

        # Either load or calculate entity embeddings
        if os.path.isfile(self.entity_embedding_path):
            with open(self.entity_embedding_path, 'r', encoding='utf-8') as json_file:
                self.entity_embeddings = json.load(json_file)
        else:
            self.entity_embeddings = _calculate_entity_embeddings(selection)

        # Put all entity embeddings with the same type into one numpy array
        index_to_string = {type_: {} for type_ in self.entity_embeddings}
        for type_ in self.entity_embeddings:
            embeddings = []
            for instance in self.entity_embeddings[type_]:
                index_to_string[type_][len(embeddings)] = instance
                embeddings.append(self.entity_embeddings[type_][instance])
            embeddings = [np.zeros(768)] if not embeddings else embeddings
            type_array = np.stack(embeddings)
            self.type_arrays[type_] = type_array

def get_parser():
    parser = argparse.ArgumentParser(description='Create a distantly supervised dataset of scientific documents')
    parser.add_argument('--ontology_entities_path', type=str, default="data/ontology/ontology_entities.csv",
                        help="path to the ontology entities file")
    parser.add_argument('--ontology_relations_path', type=str, default="data/ontology/ontology_relations.csv",
                        help="path to the ontology relations file")
    parser.add_argument('--data_path', type=str, default="data/ScientificDocuments/",
                        help='path to the folder containing scientific documents/zeta objects')
    parser.add_argument('--output_path', type=str, default="data/DistantlySupervisedDatasets/", help="output path")
    parser.add_argument('--entity_embedding_path', type=str, default="data/ontology/entity_embeddings.json",
                        help="path to file of precalculated lexical embeddings of the entities")
    parser.add_argument('--selection', type=int, nargs=2, default=None,
                        help="start and end of file range for train/test split")
    parser.add_argument('--label_function', type=int, default=2, choices=range(0, 3),
                        help="0 = string, 1 = embedding, 2 = string + embedding")
    parser.add_argument('--timestamp_given', default=False, action="store_true")
    parser.add_argument('--cos_theta', type=float, default=0.83,
                        help="similarity threshold for embedding based labeling")    
    return parser    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    dataset = DistantlySupervisedDatasets(args.ontology_entities_path, args.ontology_relations_path, args.data_path,
                                         args.entity_embedding_path, args.output_path, args.timestamp_given, args.cos_theta)
    dataset.create(label_function=args.label_function, selection=tuple(args.selection))
