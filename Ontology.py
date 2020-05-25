import read
from write import save_json, save_list, save_copy
from embedders import glue_subtokens, BertEmbedder
from collections import Counter, defaultdict
from heuristics import string_match, vote
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from argparser import get_parser
import faiss_index
from os.path import join as jp


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
        version=4,
        parent_path="data/ontology/",
        entities_file_name="ontology_entities.csv",
        relations_file_name="ontology_relations.csv",
        faiss_dir="faiss/"
    ):
        self.version = version
        self.parent_path = jp(parent_path, "v{}/".format(version))
        self.entities_file_name = entities_file_name
        self.relations_file_name = relations_file_name
        self.faiss_dir = faiss_dir
        self.entities = read.read_ontology_entity_types(jp(self.parent_path, self.entities_file_name))
        self.entity_index, self.entity_table = None, None
        self.entity_types = None
        self.relations = read.read_ontology_relation_types(jp(self.parent_path, self.relations_file_name))
        self.types = self.convert_ontology_types()

    def calculate_entity_embeddings(self, data_iterator, embedder, token_pooling="none", mention_pooling="none"):
        def _accumulate_mean(embedding, tokens, full_term, type_, ontology_embeddings):
            entry = {"type": type_, "string": tokens[0], "full_term": full_term}
            embedding = torch.stack(embeddings).mean()
            old_mean = ontology_embeddings[full_term]["embeddings"][0]
            n = ontology_embedding = ontology_embeddings[full_term]["count"]
            new_mean = old_mean + ((embedding-old_mean)/n)
            ontology_embeddings[full_term]["embeddings"][0] = new_mean
            ontology_embeddings[full_term]["entries"][0] = entry

        def _accumulate_max(embedding, tokens, full_term, type_, ontology_embeddings):
            entry = {"type": type_, "string": tokens[0], "full_term": full_term}
            old_max = ontology_embeddings[full_term]["embeddings"][0]
            t = torch.stack(embeddings+[old_max])
            new_max, _ = t.max(dim=0)
            ontology_embeddings[full_term]["embeddings"][0] = new_max
            ontology_embeddings[full_term]["entries"][0] = entry

        def _accumulate_absmax(embedding, tokens, full_term, type_, ontology_embeddings):
            entry = {"type": type_, "string": tokens[0], "full_term": full_term}
            old_absmax = ontology_embeddings[full_term]["embeddings"][0]
            t = torch.stack(embeddings+[old_absmax])
            abs_max_indices = torch.abs(t).argmax(dim=0)
            new_absmax = t.gather(0, abs_max_indices.view(1,-1)).squeeze()
            ontology_embeddings[full_term]["embeddings"][0] = new_absmax
            ontology_embeddings[full_term]["entries"][0] = entry

        def _accumulate_none(embedding, tokens, full_term, type_, ontology_embeddings):
            for embedding, token in zip(embeddings, tokens):
                entry = {"type": type_, "string": token, "full_term": full_term}
                ontology_embeddings[full_term]["embeddings"].append(embedding)
                ontology_embeddings[full_term]["entries"].append(entry)

        # Check for existing index + table
        self.entity_index, self.entity_table = faiss_index.load(jp(self.parent_path, self.faiss_dir), 
            token_pooling, mention_pooling, jp(self.parent_path, self.faiss_dir))
        if self.entity_index and self.entity_table:
            return self.entity_index, self.entity_table
        
        print("Calculating ontology entity embeddings using |{}| token pooling and |{}| mention pooling...".format(
            token_pooling, mention_pooling))
        self.entity_index = faiss_index.init(embedder.embedding_size)
        self.entity_table = []
        f_accumulation = {"mean":_accumulate_mean, "absmax":_accumulate_absmax, "max":_accumulate_max,
            "none": _accumulate_none}
        zeros = torch.zeros(embedder.embedding_size)
        zeros_entry = {"type": "null", "string": "null"}
        ontology_embeddings = defaultdict(lambda : {"embeddings":[zeros], "entries":[zeros_entry], "count": 0})

        # Accumulate entity embeddings
        for sentence_subtokens, sentence_embeddings, _ in data_iterator.iter_sentences():
            glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)
            string_matches, matched_strings = string_match(glued_tokens, self, embedder)
            for i, (span_start, span_end, type_) in enumerate(string_matches):
                embeddings, matched_tokens = embedder.reduce_embeddings(sentence_embeddings, 
                    span_start, span_end, sentence_subtokens, glued2tok, token_pooling)
                entity_string = matched_strings[i]
                ontology_embeddings[entity_string]["count"] += 1
                f_accumulation.get(mention_pooling)(embeddings, matched_tokens, entity_string, 
                    type_, ontology_embeddings) 

        # Index entities
        embeddings = []
        entries = []
        for d in ontology_embeddings.values():
            embeddings += [emb.tolist() for emb in d["embeddings"]]
            entries += d["entries"]
        data = np.array(embeddings, dtype="float32")
        data_norm = preprocessing.normalize(data, norm="l2")    
        self.entity_index.add(data_norm)
        self.entity_table += entries

        # Save
        faiss_index.save(self.entity_index, self.entity_table, token_pooling, mention_pooling, jp(self.parent_path, self.faiss_dir))

        return self.entity_index, self.entity_table


    def fetch_entity(self, i):
        entity = self.entity_table[i]
        entity_string = entity.get("string")
        entity_full_term = entity.get("full_term")
        entity_type = entity.get("type")

        return entity_type, entity_string, entity_full_term


    def evaluate_entity_embeddings(self, data_iterator, embedder, token_pooling="mean"):
        print("Calculating |{}| embedding similarity of identical strings...".format(token_pooling))
        entity_similarity_scores = defaultdict(list)
        for sentence_subtokens, sentence_embeddings, _ in data_iterator.iter_sentences():
            glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)
            string_matches, matched_strings = string_match(glued_tokens, self, embedder)
            for i, (start, end, type_) in enumerate(string_matches):
                embeddings, emb_tokens = embedder.reduce_embeddings(sentence_embeddings, start, end, glued_tokens, glued2tok, token_pooling)
                # print(glued_tokens)
                entity_string = matched_strings[i]
                mentioned_embeddings = torch.stack(embeddings).numpy()
                D, I = self.entity_index.search(mentioned_embeddings, 1)
                t, vt, vs, vft = vote(D.reshape(len(D)), I.reshape(len(D)), self)
                if entity_string in vft:
                    entity_similarity_scores[entity_string].append(int(D.mean())) 
                else:
                    print(entity_string, vft)
        
        entity_means = [np.array(v).mean() for k, v in entity_similarity_scores.items() if v]
        overall_mean = np.array(entity_means).mean()
        print("Average distance over all concepts for |{}| token pooling: {:0.2f}".format(token_pooling, overall_mean))

        return entity_similarity_scores

    def convert_ontology_types(self):
        types = {}
        entities_df = pd.read_csv(jp(self.parent_path, self.entities_file_name))
        relations_df = pd.read_csv(jp(self.parent_path, self.relations_file_name))
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

        save_json(types, jp(self.parent_path, 'ontology_types.json'))

        return types


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()




