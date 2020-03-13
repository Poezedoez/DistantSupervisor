import json
from embedders import BertEmbedder
import os
import glob
import numpy as np
from pathlib import Path
import time
from utils import KnuthMorrisPratt
from collections import defaultdict, Counter
import argparse
import csv
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from pretty_print import pretty_write


def _read_ontology_entities(path):
    ontology_entities = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip headers
        for _, class_, instance in csv_reader:
            ontology_entities[class_].append(instance)

    return ontology_entities


def _read_ontology_relations(path):
    ontology_relations = {}
    with open(path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # skip headers
        for _, head, _, _ in csv_reader:
            ontology_relations[head] = {}
        for _, head, relation, tail in csv_reader:
            ontology_relations[head][tail] = relation

    return ontology_relations


def _glue_subtokens(subtokens, includes_special_tokens=True):
    tokens = subtokens[1:-1] if includes_special_tokens else subtokens
    glued_tokens = []
    tok2glued = []
    glued2tok = []
    for i, token in enumerate(tokens):
        if token.startswith('##'):
            glued_tokens[len(glued_tokens) - 1] = glued_tokens[len(glued_tokens) - 1] + token.replace('##', '')
        else:
            glued2tok.append(i)
            glued_tokens.append(token)

        tok2glued.append(len(glued_tokens) - 1)

    return glued_tokens, tok2glued, glued2tok


class DistantlySupervisedDataset:
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
        class_arrays (dict): dict of instance embeddings stacked for one class
        flist (list): list of files used
        dataset (list): list of annotated sentence datapoints
        
    """

    def __init__(
            self,
            ontology_entities_path="data/ontology_entities.csv",
            ontology_relations_path="data/ontology_relations.csv",
            document_path="data/ScientificDocuments/",
            entity_embedding_path="data/entity_embeddings.json",
            output_path="data/DistantlySupervisedDatasets/"
    ):

        self.ontology_entities = _read_ontology_entities(ontology_entities_path)
        self.ontology_relations = _read_ontology_relations(ontology_relations_path)
        self.embedder = BertEmbedder('data/scibert_scivocab_cased')
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.document_path = document_path
        self.entity_embedding_path = entity_embedding_path
        self.output_path = output_path + '{}/'.format(self.timestamp)
        self.statistics = {"relations": Counter(), "relations_total": 0,
                           "entities": {class_: Counter() for class_ in self.ontology_entities},
                           "entity_sentences": 0, "sentences_processed": 0, "entities_total": 0, "tokens_total": 0,
                           "relation_candidates": 0}
        self.class_arrays = {}
        self.flist = []
        self.dataset = []

    def create(self, verbose=True, label_function=0, selection=None):
        if label_function > 0:
            self._load_class_arrays()
        start_time = time.time()
        for sentence_subtokens, document_embeddings, offset in self._iter_sentences(selection):
            self._label_sentence(sentence_subtokens, document_embeddings, offset, label_function)
        end_time = time.time()
        self.statistics["time_taken"] = int(end_time - start_time)
        self._save()
        if verbose:
            self.print_statistics()

    def _save(self):
        directory = os.path.dirname(self.output_path)
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Save dataset
        dataset_path = self.output_path + 'dataset.json'
        with open(dataset_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.dataset, json_file)

        # Save statistics
        with open(self.output_path + 'statistics.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.statistics, json_file)

        # Save ontology used
        shutil.copyfile(args.ontology_entities_path, self.output_path + 'ontology_entities.csv')
        shutil.copyfile(args.ontology_relations_path, self.output_path + 'ontology_relations.csv')

        # Save list of documents used for the set
        with open(self.output_path + 'filelist.txt', 'w', encoding='utf-8') as txt_file:
            for file in self.flist:
                txt_file.write("{} \n".format(file))

        # Save pretty output of dataset
        pretty_write(dataset_path, self.output_path+'pretty_output.txt')

    def print_statistics(self, statistics=None):
        if statistics:
            with open(statistics, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        else:
            stats = self.statistics

        print("--- STATISTICS ---")
        print("Processed {} sentences of which {} contained at least one entity".format(
            stats["sentences_processed"], stats["entity_sentences"]
        ))
        print("Time taken: {} seconds".format(stats["time_taken"]))
        print("--- Entities ---")
        tokens_per_entity = stats["tokens_total"] / stats["entities_total"] if(
            stats["entities_total"] != 0
        ) else 0
        print("Every {} tokens an entity occurs".format(tokens_per_entity))
        print("Entities were found in the following classes:")
        for class_, instance_counter in stats["entities"].items():
            count = sum([count for _, count in instance_counter.items()])
            print(class_, count)
        print("The most frequently labeled entities per class are:")
        for class_, instance_counter in stats["entities"].items():
            print("{} \t".format(class_), Counter(instance_counter).most_common(5))
        print("--- Relations ---")
        relations_per_sentence = stats["relation_candidates"]/stats["relations_total"] if(
            stats["relations_total"] != 0
        ) else 0
        print("Every {} sentences with at least two entities a relation occurs".format(relations_per_sentence))
        print("Relations were found in the following classes:")
        for relation, count in stats["relations"].items():
            print(relation, count)

    def _iter_sentences(self, selection=None):
        for document_sentences, document_embeddings in self._read_documents(selection):
            offset = 0
            for sentence in document_sentences:
                yield sentence, document_embeddings, offset
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

    def _string_match(self, tokens, string):
        tokenized_string = [token.lower() for token in self.embedder.tokenize(string)]
        glued_string, _, _ = _glue_subtokens(tokenized_string, includes_special_tokens=False)
        tokens = [token.lower() for token in tokens]
        string_length = len(glued_string)
        matches = [(occ, occ + string_length) for occ in KnuthMorrisPratt(tokens, glued_string)]

        return matches

    def _knn_match(self, sentence_embeddings, tok2glued, class_, threshold=0.9):
        matches = []
        prev_entity = False
        start = 0
        similarities = cosine_similarity(sentence_embeddings, self.class_arrays[class_])
        max_similarities = similarities.max(axis=1)
        for token in tok2glued:
            score = max_similarities[token]
            # entity span starts
            if score > threshold and not prev_entity:
                start = token
                prev_entity = True
                continue

            # entity span continues
            elif score > threshold and prev_entity:
                prev_entity = True
                continue

            # etity span ends
            elif prev_entity:
                matches.append((tok2glued[start], tok2glued[token]))

            start = token
            prev_entity = False

        return matches

    def _label_sentence(self, sentence_subtokens, document_embeddings, offset, label_function=0):
        def _label_relations(entities):
            relations = []
            if len(entities) > 1:
                self.statistics["relation_candidates"] += 1
                pairs = [(a, b) for a in range(0, len(entities)) for b in range(0, len(entities))]
                for head, tail in pairs:
                    relation = self.ontology_relations.get(entities[head]["type"], {}).get(entities[tail]["type"])
                    if relation:
                        self.statistics["relations_total"] += 1
                        self.statistics["relations"][relation] += 1
                        relations.append({"type": relation, "head": head, "tail": tail})
            return relations

        def _label_entities(tokens, tok2glued):
            entities = []
            sentence_embeddings = document_embeddings[offset+1: offset+len(tokens)+1]  # skip [CLS] and [SEP]
            for class_, string_instances in self.ontology_entities.items():
                for string_instance in string_instances:
                    if label_function == 0 or label_function == 2:
                        string_matches = self._string_match(glued_tokens, string_instance)
                    else:
                        string_matches = []
                    if label_function == 1 or label_function == 2:
                        knn_matches = self._knn_match(sentence_embeddings, tok2glued, class_)
                    else:
                        knn_matches = []
                    matches = set(string_matches + knn_matches)
                    for start, end in matches:
                        entity_string = " ".join(glued_tokens[start:end]).lower()
                        self.statistics["entities"][class_][entity_string] += 1
                        entities.append({"type": class_, "start": start, "end": end})
            return entities

        glued_tokens, tok2glued, _ = _glue_subtokens(sentence_subtokens)
        entities = _label_entities(glued_tokens, tok2glued)
        relations = _label_relations(entities)
        self.statistics["sentences_processed"] += 1

        if not entities:
            return

        self.statistics["entity_sentences"] += 1
        self.statistics["tokens_total"] += len(glued_tokens)
        self.statistics["entities_total"] += len(entities)
        joint_string = "".join(glued_tokens)
        hash_string = hash(joint_string)
        training_instance = {"tokens": glued_tokens, "entities": entities,
                             "relations": relations, "orig_id": hash_string}
        self.dataset.append(training_instance)

    def _load_class_arrays(self):
        def _calculate_entity_embeddings():
            # Sum all entity instances
            entity_embeddings = {class_: defaultdict(lambda: np.zeros(768)) for class_ in self.ontology_entities}
            entity_counter = {class_: Counter() for class_ in self.ontology_entities.keys()}
            for sentence_subtokens, document_embeddings, offset in self._iter_sentences(selection=args.selection):
                glued_tokens, _, _ = _glue_subtokens(sentence_subtokens)
                sentence_embeddings = document_embeddings[offset:offset + len(sentence_subtokens)]
                for class_, string_instances in self.ontology_entities.items():
                    for string_instance in string_instances:
                        string_matches = self._string_match(glued_tokens, string_instance)
                        for start, end in string_matches:
                            embedding = np.stack(sentence_embeddings[start:end]).mean(axis=0)
                            # embedding = sentence_embeddings[start]
                            token = string_instance.lower()
                            entity_embeddings[class_][token] += embedding
                            entity_counter[class_][token] += 1

            # Average the sum of embeddings
            for class_, count_dict in entity_counter.items():
                for token, count in count_dict.items():
                    summed_embedding = entity_embeddings[class_][token]
                    entity_embeddings[class_][token] = (summed_embedding / count).tolist()

            return entity_embeddings

        if os.path.isfile(self.entity_embedding_path):
            with open(self.entity_embedding_path, 'r', encoding='utf-8') as json_file:
                entity_embeddings = json.load(json_file)
        else:
            entity_embeddings = _calculate_entity_embeddings()
            with open(self.entity_embedding_path, 'w', encoding='utf-8') as json_file:
                json.dump(entity_embeddings, json_file)

        for class_ in entity_embeddings:
            embeddings = [entity_embeddings[class_][instance] for instance in entity_embeddings[class_]]
            embeddings = [np.zeros(768)] if not embeddings else embeddings
            class_array = np.stack(embeddings)
            self.class_arrays[class_] = class_array


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
    parser.add_argument('--label_function', type=int, default=0, choices=range(0, 3),
                        help="0 = string, 1 = knn, 2 = string + knn")
    args = parser.parse_args()
    dataset = DistantlySupervisedDataset(args.ontology_entities_path, args.ontology_relations_path, args.document_path,
                                         args.entity_embedding_path, args.output_path)
    dataset.create(label_function=args.label_function, selection=tuple(args.selection))
