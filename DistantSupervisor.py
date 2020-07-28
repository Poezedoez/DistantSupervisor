import json
from embedders import BertEmbedder, glue_subtokens
import numpy as np
import time
from collections import defaultdict, Counter
from read import DataIterator
from write import save_json, save_list, save_copy, print_dataset, print_statistics
from heuristics import EntityMatcher, RelationMatcher
from Ontology import Ontology
import copy
from argparser import get_parser


class DistantSupervisor:
    """
    Args:
        data_path (str): path to the folder of scientific documents containing
            named with document id's
        ontology_path (str): path the parent directory of ontology files
        tokenizer_path (str): path to tokenizer
        output_path (str): path to store results
        timestamp_given (bool): whether a time stamp is included in the output path
        cos_theta (float): similarity threshold for embedding labeling
        filter_sentences (bool): whether to filter dirty sentences from the data
        token_pooling (str): name of the pooling function for token embeddings
        mention_pooling (str): name of the pooling function for mention embeddings

    Attr:
        ontology (Ontology): ontology with entities (+embeddings) and relations
        embedder (Embedder): embedder used for tokenization and obtaining embeddings
        timestamp (str): string containing object creation time
        data_path (str): stored document path from init argument
        output_path (str): stored output path from init argument combined with timestamp
        label_statistics (dict): dict to store statistics per label function
        global_statistics (dict): dict to store shared statistics between different label functions
        flist (list): list of files used
        label_strategies (dict): mapping of label function (int) to its name (str)
        datasets (dict): list of annotated sentence datapoints
        
    """

    def __init__(
        self,
        data_path="data/ScientificDocuments/",
        ontology_path="data/ontology/v4/",
        tokenizer_path="data/scibert_scivocab_cased",
        output_path="data/DistantlySupervisedDatasets/",
        timestamp_given=False,
        cos_theta=0.80,
        filter_sentences=True,
        token_pooling="mean",
        mention_pooling="mean",
        entity_fraction=1.0
    ):
        self.ontology = Ontology(ontology_path, entity_fraction=entity_fraction)
        self.embedder = BertEmbedder(tokenizer_path)
        self.timestamp = '' if timestamp_given else time.strftime("%Y%m%d-%H%M%S")+'/'
        self.data_path = data_path
        self.filter_sentences = filter_sentences
        self.token_pooling = token_pooling
        self.mention_pooling = mention_pooling
        self.output_path = output_path + self.timestamp
        self.cos_theta = cos_theta
        self.flist = set()
        self.label_strategies = {0: "string_labeling", 1: "embedding_labeling", 2: "combined_labeling"}
        self.datasets = {"string_labeling": [], "embedding_labeling": [], "combined_labeling": []}
        self.label_statistics, self.global_statistics = self._prepare_statistics()
        self.entity_matcher = EntityMatcher(self.ontology, self.embedder, self.token_pooling, cos_theta)
        self.relation_matcher = RelationMatcher(self.ontology)
        self.entity_fraction = entity_fraction
        

    def supervise(self, label_strategy=0, selection=None):
        # print("Number of processors available to use:", len(os.sched_getaffinity(0)))
        start_time = time.time()
        print("Creating dataset...")

        # Init data iterator
        iterator = DataIterator(
            self.data_path, 
            selection=selection, 
            includes_special_tokens=True, 
            filter_sentences=self.filter_sentences
        )

        # Ready ontology embeddings
        if label_strategy > 0:
            self.ontology.calculate_entity_embeddings(iterator, self.embedder, self.token_pooling, self.mention_pooling)
                
        # Supervise sentences        
        for sentence_subtokens, sentence_embeddings, doc_name in iterator.iter_sentences():
            self._label_sentence(sentence_subtokens, sentence_embeddings, label_strategy)
            self.flist.add(doc_name)

        end_time = time.time()
        self.global_statistics["time_taken"] = int(end_time - start_time)
        self._save()


    def _save(self):       
        # Save datasets of different labeling functions
        for label_function, dataset in self.datasets.items():
            if self.label_statistics[label_function]["skip"]: # skip empty dataset
                continue
            dataset_path = self.output_path + '{}/dataset.json'.format(label_function)
            save_json(dataset, dataset_path)

            # Save dataset statistics
            statistics_path = self.output_path + '{}/statistics.json'.format(label_function)
            self.global_statistics["label_function"] = label_function
            self.label_statistics[label_function].update(self.global_statistics)
            save_json(self.label_statistics[label_function], statistics_path)
            print_statistics(statistics_path)
            
            # Save pretty output of labeled examples
            print_dataset(dataset_path, self.output_path+'{}/classified_examples.txt'.format(label_function))

        # Save list of selected documents used for the split
        save_list(self.flist, self.output_path+'filelist.txt')

    
    def _prepare_statistics(self):
        entity_types = set(self.ontology.entities.values())
        label_statistics = {"relations": Counter(), "relations_total": 0,
                    "entities": {type_: Counter() for type_ in entity_types},
                    "entity_sentences": 0, "entities_total": 0, "tokens_total": 0,
                    "relation_candidates": 0, "skip":True}
        global_statistics = {}
        global_statistics["sentences_processed"] = 0
        global_statistics["cos_theta"] = self.cos_theta
        global_statistics["ontology_path"] = self.ontology.parent_path

        return {dataset: copy.deepcopy(label_statistics) for dataset in self.datasets}, global_statistics


    def _label_sentence(self, sentence_subtokens, sentence_embeddings, label_function=0):
        def label_relations(entities, glued_tokens):
            relations = []
            if len(entities) < 2:
                return relations
            
            # Get type pairs
            relations += self.relation_matcher.pair_match(entities)

            # Get isA relations
            relations += self.relation_matcher.pattern_match(glued_tokens, entities)

            return relations

        def label_entities(matches):
            entities = []
            for start, end, type_ in matches:
                entities.append({"type": type_, "start": start, "end": end})

            return entities

        glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)

        # Find string entity matches
        do_string_matching = (label_function == 0 or label_function == 2)
        string_matches, _ = self.entity_matcher.string_match(glued_tokens, do_string_matching)
        string_entities = label_entities(string_matches)
        string_relations = label_relations(string_entities, glued_tokens)

        # Find embedding entity matches
        do_embedding_matching = (label_function == 1 or label_function == 2)
        embedding_matches = self.entity_matcher.embedding_match(sentence_embeddings, sentence_subtokens, 
                                                                glued2tok, glued_tokens, do_embedding_matching)
        embedding_entities = label_entities(embedding_matches)
        embedding_relations = label_relations(embedding_entities, glued_tokens)

        # Find combined entity matches
        do_combined_matching = (label_function == 2)
        combined_matches = self.entity_matcher.combined_match(string_matches, embedding_matches, do_combined_matching)
        combined_entities = label_entities(combined_matches)
        combined_relations = label_relations(combined_entities, glued_tokens)
        
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


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    supervisor = DistantSupervisor(
        data_path=args.data_path, 
        ontology_path=args.ontology_path, 
        output_path=args.output_path, 
        timestamp_given=args.timestamp_given, 
        cos_theta=args.cos_theta, 
        filter_sentences=args.filter_sentences, 
        token_pooling=args.token_pooling,
        mention_pooling=args.mention_pooling,
        entity_fraction=args.entity_fraction
    )

    supervisor.supervise(label_strategy=args.label_strategy, selection=tuple(args.selection))
