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


class Ontology:
    """
    Args:
        entities_path (str): path to the ontology entities csv file
        relations_path (str): path to the ontology relations csv file
        entity_embedding_path (str): path to the precalculated entity embeddings of the ontology

    Attr:
        ontology_entities (dict): ontology loaded with json from the given path
        ontology_relations (dict): ontology loaded with json from the given path
        entity_embedding_path (str): stored entity embedding path from init argument
        embedding_array (dict): dict of instance embeddings stacked for one entity type
        
    """

    def __init__(
        self,
        entities_path="data/ontology/ontology_entities.csv",
        relations_path="data/ontology/ontology_relations.csv",
        entity_embedding_path="data/ontology/entity_embeddings.json",
    ):

        self.entities_path = entities_path
        self.relations_path = relations_path
        self.entity_embedding_path = entity_embedding_path
        self.entities = read.read_ontology_entity_types(entities_path)
        self.entity_embeddings = read.read_entity_embeddings(entity_embedding_path)
        self.embedding_array, self.array2entity = self.embeddings_to_array()
        self.relations = read.read_ontology_relation_types(relations_path)
        self.types = self.convert_ontology_types()
        
            
    def calculate_entity_embeddings(self, data_iterator, embedder, f_reduce="mean"):
        def _accumulate_mean(embedding, entity_string, entity_embeddings):
            entity_embeddings[entity_string]["embedding"] += embedding
            entity_embeddings[entity_string]["count"] += 1

        def _accumulate_max(embedding, entity_string, entity_embeddings):
            ontology_embedding = entity_embeddings[entity_string]["embedding"]
            t = torch.stack([embedding]+[ontology_embedding])
            max_embedding, _ = t.max(dim=0)
            entity_embeddings[entity_string]["embedding"] = max_embedding
            entity_embeddings[entity_string]["count"] += 1

        def _accumulate_absmax(embedding, entity_string, entity_embeddings):
            ontology_embedding = entity_embeddings[entity_string]["embedding"]
            t = torch.stack([embedding]+[ontology_embedding])
            abs_max_indices = torch.abs(t).argmax(dim=0)
            absmax_embedding = t.gather(0, abs_max_indices.view(1,-1)).squeeze()
            entity_embeddings[entity_string]["embedding"] = absmax_embedding
            entity_embeddings[entity_string]["count"] += 1

        print("Calculating ontology entity embeddings...")
        entity_embeddings = {token: {"embedding": torch.zeros(embedder.embedding_size), "count": 0}  for token in self.entities}
        for sentence_subtokens, sentence_embeddings, _ in data_iterator.iter_sentences():
            glued_tokens, _, glued2tok = glue_subtokens(sentence_subtokens)
            string_matches, matched_strings = string_match(glued_tokens, self, embedder)
            for i, (start, end, type_) in enumerate(string_matches):
                emb, emb_tokens = embedder.reduce_embeddings(sentence_embeddings, start, end, glued_tokens, glued2tok, f_reduce)
                entity_string = matched_strings[i]
                assert(entity_string in self.entities) # sanity check
                f_accumulation = {"mean":_accumulate_mean, "abs_max":_accumulate_absmax, "max":_accumulate_max}
                f_accumulation.get(f_reduce)(emb, entity_string, entity_embeddings)

        # Convert to list (and average in case of mean reduction)
        for entity, entry in entity_embeddings.items():
            embedding, count = entry["embedding"], entry["count"]
            if f_reduce == "mean":
                embedding = embedding / count if count else torch.zeros(embedder.embedding_size)
            entity_embeddings[entity]["embedding"] = embedding.tolist()

        self.entity_embeddings = entity_embeddings

        return entity_embeddings
    

    def embeddings_to_array(self, null_shape=768):
        if not self.entity_embeddings:
            return None, None
        
        array2entity = []
        embeddings = []
        
        # Backwards compatibility :)
        try:
            for entity in self.entities:
                entry = self.entity_embeddings.get(entity, entity)
                embedding = entry.get("embedding", np.zeros(null_shape))
                embeddings.append(embedding)
                # self.entity_embeddings[entity]["embedding"] = embedding
                self.entities[entity]["count"] = entry.get("count", 0)
                array2entity.append(entity)
        except:
            print("attack of the guerilla monkeys")
            print("old entity embedding format found, converting...")
            flattened_entity_embeddings = {token: v for _, token_dict  in self.entity_embeddings.items() 
                                            for token, v in token_dict.items()}
            print(flattened_entity_embeddings.keys())
            self.entity_embeddings = {token: {"embedding": torch.zeros(null_shape), "count": 0}  
                                      for token in self.entities}
            for entity in self.entities:
                embedding = flattened_entity_embeddings.get(entity, np.zeros(null_shape))
                self.entity_embeddings[entity]["embedding"] = embedding
                embeddings.append(embedding)
                self.entities[entity]["count"] = 0
                self.entity_embeddings[entity]["count"] = 0
                array2entity.append(entity)

        # Use numpy array for sklearn cosine similarity
        embedding_array = np.stack(embeddings)
        self.embedding_array = embedding_array
        self.array2entity = array2entity
        
        return embedding_array, array2entity


    def fetch_entity(self, i):
        entity_string = self.array2entity[i]
        entity = self.entities.get(entity_string, {})
        type_ = entity.get("type", None)
        count = entity.get("count", -1)

        return entity_string, type_, count


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
                ontology_embedding = np.expand_dims(np.array(self.entity_embeddings[entity_string]["embedding"]), axis=0)
                mentioned_embedding = np.expand_dims(emb.numpy(), axis=0)
                score = cosine_similarity(mentioned_embedding, ontology_embedding)[0][0]
                # print(entity_string, score)
                entity_similarity_scores[entity_string].append(score)
                assert(entity_string in self.entities) # sanity check
        
        entity_means = [np.array(v).mean() for k, v in entity_similarity_scores.items() if v]
        overall_mean = np.array(entity_means).mean()
        filter_used = "filtered" if data_iterator.filter_sentences else "unfiltered" 
        print("Similarity score over all concepts for |{}| reduction |{}|: {:0.2f}".format(f_reduce, filter_used, overall_mean))

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
        
        return types


    def save(self, output_path):
        save_copy(self.entities_path, output_path+'ontology_entities.csv')
        save_copy(self.relations_path, output_path+'ontology_relations.csv')
        save_json(self.entity_embeddings, output_path+'entity_embeddings.json')
        save_json(self.types, output_path+'ontology_types.json')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()




