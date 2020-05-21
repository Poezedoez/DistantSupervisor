import read
from write import save_json, save_list, save_copy
from embedders import glue_subtokens, BertEmbedder
from collections import Counter, defaultdict
from heuristics import string_match
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from argparser import get_parser
import faiss_index


class Ontology:
    """
    Args:
        entities_path (str): path to the ontology entities csv file
        relations_path (str): path to the ontology relations csv file
        faiss_index_path (str): path to the folder with faiss index and table

    Attr:
        ontology_entities (dict): ontology loaded with json from the given path
        ontology_relations (dict): ontology loaded with json from the given path
        faiss_index_path (str): path to the folder with faiss index and table
        entity_index (Faiss index): faiss index with ontology entity embeddings
        entity_table (array): ordered table with embedding properties
        
    """

    def __init__(
        self,
        entities_path="data/ontology/ontology_entities.csv",
        relations_path="data/ontology/ontology_relations.csv",
        faiss_index_path="data/ontology/faiss/",
    ):

        self.entities_path = entities_path
        self.relations_path = relations_path
        self.faiss_index_path = faiss_index_path
        self.entities = read.read_ontology_entity_types(entities_path)
        self.entity_index, self.entity_table = faiss_index.load(faiss_index_path, "entities")
        self.entity_types, self.entity_types_map = None, None
        self.relations = read.read_ontology_relation_types(relations_path)
        self.types = self.convert_ontology_types()
        
            
    def calculate_entity_embeddings(self, data_iterator, embedder, f_reduce="none"):
        print("Calculating ontology entity embeddings...")
        self.entity_index = faiss_index.init(embedder.embedding_size)
        self.entity_table = []
        ontology_embeddings = []
        for sentence_subtokens, sentence_embeddings, _ in data_iterator.iter_sentences():
            glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)
            string_matches, matched_strings = string_match(glued_tokens, self, embedder)
            for i, (span_start, span_end, type_) in enumerate(string_matches):
                embedding, matched_tokens = embedder.reduce_embeddings(sentence_embeddings, 
                    span_start, span_end, glued_tokens, glued2tok, f_reduce)
                ontology_embeddings.append(embedding)
                self.entity_table.append({"type": type_, "string": matched_strings[i]})

                # sanity check
                # print(matched_strings[i])
                entity_string = matched_strings[i]
                assert(entity_string in self.entities) 

        self.entity_index.add(np.stack(ontology_embeddings))

        return self.entity_index, self.entity_table


    def fetch_entity(self, i):
        entity = self.entity_table[i]
        entity_string = entity.get("string")
        entity_type = entity.get("type")

        return entity_string, entity_type


    def evaluate_entity_embeddings(self, data_iterator, embedder, f_reduce="mean"):
        print("Calculating |{}| embedding similarity of identical strings...".format(f_reduce))
        entity_similarity_scores = defaultdict(list)
        for sentence_subtokens, sentence_embeddings, _ in data_iterator.iter_sentences():
            glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)
            string_matches, matched_strings = string_match(glued_tokens, self, embedder)
            for i, (start, end, type_) in enumerate(string_matches):
                emb, emb_tokens = embedder.reduce_embeddings(sentence_embeddings, start, end, glued_tokens, glued2tok, f_reduce)
                # print(glued_tokens)
                entity_string = matched_strings[i]
                mentioned_embedding = np.expand_dims(emb.numpy(), axis=0)
                D, I = self.entity_index.search(mentioned_embedding, 1)
                s, t = self.fetch_entity(I[0][0])
                if entity_string==s:
                    entity_similarity_scores[entity_string].append(D[0][0]) 
                else:
                    print(entity_string, s)
        
        entity_means = [np.array(v).mean() for k, v in entity_similarity_scores.items() if v]
        overall_mean = np.array(entity_means).mean()
        filter_used = "filtered" if data_iterator.filter_sentences else "unfiltered" 
        print("Average distance over all concepts for |{}| reduction |{}|: {:0.2f}".format(f_reduce, filter_used, overall_mean))

        return entity_similarity_scores

    def convert_ontology_types(self):
        types = {}
        entities_df = pd.read_csv(self.entities_path)
        relations_df = pd.read_csv(self.relations_path)
        types["entities"] = {type_:{"short": type_, "verbose": type_} for type_ in set(entities_df["Class"])}
        types["relations"] = {}
        for _, row in relations_df.iterrows():
            type_ = row["relation"]
            types["relations"][type_] = {"short": type_, "verbose": type_, "symmetric":row["symmetric"]}
        
        entity_types = {}
        entity_types_map = []
        for i, entry in enumerate(types["entities"]):
            types[entry] = i
            entity_types_map.append(entry)

        self.entity_types = entity_types
        self.entity_types_map = entity_types_map

        return types


    def save(self, output_path, f_reduce="mean", filtered=True):
        save_copy(self.entities_path, output_path+'ontology_entities.csv')
        save_copy(self.relations_path, output_path+'ontology_relations.csv')
        filter_option = "filtered" if filtered else "unfiltered"
        faiss_index.save(self.entity_index, self.entity_table, "entities_{}_{}".format(f_reduce, filter_option), 
            output_path+"faiss/")
        save_json(self.types, output_path+'ontology_types.json')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()




