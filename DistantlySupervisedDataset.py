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

class DistantlySupervisedDataset():
    '''
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
        
    '''
    def __init__(self, ontology_path="data/ontology.csv", document_path="data/ScientificDocuments/", 
        entity_embedding_path="data/entity_embeddings.json", output_path="data/DistantlySupervisedDatasets/"):
        self.ontology = self._read_ontology(ontology_path)
        self.embedder = BertEmbedder('data/scibert_scivocab_cased')
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.document_path = document_path
        self.entity_embedding_path = entity_embedding_path
        self.output_path = output_path+'{}/'.format(self.timestamp)
        self.statistics = {"classes":Counter(), "tokens":{class_:Counter() for class_ in self.ontology}, "sentences_useful":0,
                           "sentences_processed":0, "entities_total":0, "tokens_total":0}
        self.class_arrays = {}
        self.flist = []
        self.dataset = []

    def create(self, verbose=True, knn_labeling=False, selection=None):
        if knn_labeling:
            self._calculate_entity_embeddings()
        start_time = time.time()
        for sentence_subtokens, document_embeddings, offset in self._iter_sentences(selection):
            self._label_sentence(sentence_subtokens, document_embeddings, offset, knn_labeling)
        end_time = time.time()
        self.statistics["time_taken"] = end_time-start_time
        self._save()
        if verbose:
            self.print_statistics()

    def _save(self):
        directory = os.path.dirname(self.output_path)
        Path(directory).mkdir(parents=True, exist_ok=True)

        ## Save dataset
        with open(self.output_path+'dataset.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.dataset, json_file) 

        ## Save statistics
        with open(self.output_path+'statistics.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.statistics, json_file) 

        ## Save ontology used
        shutil.copyfile(args.ontology_path, self.output_path+'ontology.csv')

        ## Save list of documents used for the set
        with open(self.output_path+'filelist.txt', 'w', encoding='utf-8') as txt_file:
            for file in self.flist:
                txt_file.write("{} \n".format(file))

    def print_statistics(self, statistics=None):
        if statistics:
            with open(statistics, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        else:
            stats = self.statistics
         
        print("--- STATISTICS ---")
        print("Processed {} sentences of which {} contained at least one entity".format(
            stats["sentences_processed"], stats["sentences_useful"]
        ))
        print("Time taken: {} seconds".format(stats["time_taken"]))
        tokens_per_entity = stats["tokens_total"]/stats["entities_total"]
        print("Every {} tokens an entity occurs".format(tokens_per_entity))
        print("The following classes were found:")
        for class_, count in stats["classes"].items():
            print(class_, count)
        print("The most frequently labeled tokens per class are:")
        for class_, instance_counter in stats["tokens"].items():
            print("{} \t".format(class_), Counter(instance_counter).most_common(5))

    def _iter_sentences(self, selection=None):
        for document_sentences, document_embeddings in self._read_documents(selection):
            offset = 0
            for sentence in document_sentences:
                yield sentence, document_embeddings, offset
                offset += len(sentence)

    def _read_ontology(self, path):
        ontology = defaultdict(list)
        with open(path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None) # skip headers
            for _, class_, instance in csv_reader:
                ontology[class_].append(instance)
        
        return ontology


    def _read_documents(self, selection=None):
        path = self.document_path
        self.flist = os.listdir(path) if not selection else os.listdir(path)[selection[0]:selection[1]]
        for folder in self.flist:
            text_path = glob.glob(path+"{}/representations/".format(folder) + "text_sentences|*.tokens")[0]
            with open(text_path, 'r', encoding='utf-8') as text_json:
                text = json.load(text_json)
            embeddings_path = glob.glob(path+"{}/representations/".format(folder) + "text_sentences|*word_embeddings.npy")[0]
            embeddings = np.load(embeddings_path)

            yield text, embeddings

    def _fuse_subtokens(self, subtokens, includes_special_tokens=True):
        tokens = subtokens[1:-1] if includes_special_tokens else subtokens
        fused_tokens = []
        tok2fused = []
        fused2tok = []
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                fused_tokens[len(fused_tokens)-1] = fused_tokens[len(fused_tokens)-1] + token.replace('##', '')
            else:
                fused2tok.append(i)
                fused_tokens.append(token)
                
            tok2fused.append(len(fused_tokens)-1)

        return fused_tokens, tok2fused, fused2tok

    def _string_match(self, tokens, string):
        tokenized_string = [token.lower() for token in self.embedder.tokenize(string)]
        fused_string, _, _ = self._fuse_subtokens(tokenized_string, includes_special_tokens=False)
        tokens = [token.lower() for token in tokens]
        string_length = len(fused_string)
        matches = [(occ, occ+string_length) for occ in KnuthMorrisPratt(tokens, fused_string)]

        return matches

    def _knn_match(self, sentence_embeddings, tok2fused, class_, threshold=0.9):
        matches = []
        prev_entity = False
        start = 0
        similarities = cosine_similarity(sentence_embeddings, self.class_arrays[class_])
        max_similarities = similarities.max(axis=1)
        for i, token in enumerate(tok2fused):
            score = max_similarities[i]
            ## entity span starts
            if score > threshold and not prev_entity:
                start = i
                prev_entity = True
                continue

            ## entity span continues
            elif score > threshold and prev_entity:
                prev_entity = True
                continue

            ## etity span ends
            elif prev_entity:
                matches.append((tok2fused[start], token))

            start = i
            prev_entity = False

        return matches

        
    def _label_sentence(self, sentence_subtokens, document_embeddings, offset, knn_labeling=False):
        fused_tokens, tok2fused, _ = self._fuse_subtokens(sentence_subtokens)
        sentence_embeddings = document_embeddings[offset:offset+len(sentence_subtokens)]
        training_instance = {}
        entities = []
        for class_, string_instances in self.ontology.items():
            for string_instance in string_instances:
                string_matches = self._string_match(fused_tokens, string_instance)
                knn_matches = self._knn_match(sentence_embeddings, tok2fused, class_) if knn_labeling and string_matches else []
                matches = set(string_matches+knn_matches)
                # TODO: temp!!
                matches = knn_matches
                for start, end in matches:
                    entity_string = " ".join(fused_tokens[start:end]).lower()
                    print("knn_matched the instance |{}| to class |{}|".format(entity_string, class_))
                    print("original sentence:", " ".join(fused_tokens))
                    self.statistics["classes"][class_] += 1
                    self.statistics["tokens"][class_][entity_string] += 1
                    
                    entities.append({"type":class_, "start":start, "end":end})

        if entities:
            self.statistics["sentences_useful"] += 1
            self.statistics["tokens_total"] += len(fused_tokens)
            self.statistics["entities_total"] += len(entities)
            joint_string = "".join(fused_tokens)
            hash_string = hash(joint_string)
            training_instance = {"tokens":fused_tokens, "entities":entities, "relations":[], "orig_id":hash_string}
            self.dataset.append(training_instance)

        self.statistics["sentences_processed"] += 1

    def _calculate_entity_embeddings(self):

        if os.path.isfile(self.entity_embedding_path):
            with open(self.entity_embedding_path) as json_file:
                self.entity_embeddings = json.load(json_file) 
                return

        ## Sum all entity instances
        entity_embeddings = {class_:defaultdict(lambda: np.zeros(768)) for class_ in self.ontology}
        entity_counter = {class_:Counter() for class_ in self.ontology.keys()}
        for sentence_subtokens, document_embeddings, offset in self._iter_sentences(selection=args.selection):
            fused_tokens, _, _ = self._fuse_subtokens(sentence_subtokens)
            sentence_embeddings = document_embeddings[offset:offset+len(sentence_subtokens)]
            for class_, string_instances in self.ontology.items():
                for string_instance in string_instances:
                    string_matches = self._string_match(fused_tokens, string_instance)
                    for start, end in string_matches:
                        embedding = np.stack(sentence_embeddings[start:end]).mean(axis=0)
                        # embedding = sentence_embeddings[start]
                        token = string_instance.lower()
                        entity_embeddings[class_][token] += embedding
                        entity_counter[class_][token] +=1

        ## Average the sum of embeddings
        for class_, count_dict in entity_counter.items():
            for token, count in count_dict.items():
                summed_embedding = entity_embeddings[class_][token]
                entity_embeddings[class_][token]= summed_embedding/count
                embeddings = [entity_embeddings[class_][instance] for instance in entity_embeddings[class_]]
                embeddings = [np.zeros(768)] if not embeddings else embeddings
            class_tensor = np.stack(embeddings)
            self.class_arrays[class_] = class_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a distantly supervised dataset of scientific documents')
    parser.add_argument('--ontology_path', type=str, default="data/ontology.csv", help="path to the ontology file")
    parser.add_argument('--document_path', type=str, help='path to the folder containing scientific documents',
        default="data/ScientificDocuments/all/")
    parser.add_argument('--output_path', type=str, default="data/DistantlySupervisedDatasets/", help="output path")
    parser.add_argument('--entity_embedding_path', type=str, default="data/entity_embeddings.json", 
        help="path to file of precalculated lexical embeddings of the entities")
    parser.add_argument('--selection', type=int, nargs=2, default=None, help="start and end of file range for train/test split")
    parser.add_argument('--knn_labeling', type=int, default=0, 
        help="use knn unsupervised labeling in conjunction with default string matching")
    args = parser.parse_args()
    dataset = DistantlySupervisedDataset(args.ontology_path, args.document_path, args.output_path, args.entity_embedding_path)
    # dataset.create(knn_labeling=args.knn_labeling, selection=tuple(args.selection))
    dataset._calculate_entity_embeddings()
    dataset._save()
