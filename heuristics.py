from utils import KnuthMorrisPratt
from embedders import glue_subtokens
from nltk.translate.ribes_score import position_of_ngram
import nltk
from sklearn import preprocessing
import numpy as np
import re
import copy
from collections import Counter

def _contains_verb(tokens):
    VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    pos_tags = {t[1] for t in nltk.pos_tag(tokens)}
    has_verbs = VERBS.intersection(pos_tags)

    return has_verbs

def _alphabetical_sequence(tokens, symbol_threshold):
    alphabetical_sequence = False
    alphabet_tokens = len([s for s in tokens if re.match("^[a-zA-Z]*$", s)])
    if float(alphabet_tokens)/float(len(tokens)) > symbol_threshold:
        alphabetical_sequence = True

    return alphabetical_sequence


def proper_sequence(tokens, symbol_threshold=0.2, verbose=False):
    has_verbs = _contains_verb(tokens)
    alphabetical_sequence = _alphabetical_sequence(tokens, symbol_threshold)
    proper = (has_verbs and alphabetical_sequence)
    if not proper and verbose:
        v = "VERBS" if not has_verbs else ''
        a = "a-Z" if not alphabetical_sequence else ''
        print("The following sentence does not have {} {}".format(v, a))
    return proper


def noun_phrases(tokens):
    grammar = r"""
    NALL: {<NN>*<NNS>*<NNP>*<NNPS>*}
    NC: {<JJ>*<NALL>+}
    NP: {<NC>+}  

    """

    cp = nltk.RegexpParser(grammar)
    pos = nltk.pos_tag(tokens)
    result = cp.parse(pos)
    noun_phrases = []
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        np = ''
        for x in subtree.leaves():
            np = np + ' ' + x[0]
        noun_phrases.append(np.strip())

    selected_spans = []
    selected_nps = []
    for np in noun_phrases:
        splitted_np = np.split()
        start = position_of_ngram(tuple(splitted_np), tokens)
        end = start+len(splitted_np)
        np_tokens = tokens[start:end]
        if _alphabetical_sequence(np_tokens, symbol_threshold=0.4): 
            selected_spans.append((start, end))
            selected_nps.append(np)

    return selected_nps, selected_spans

class EntityMatcher:
    def __init__(self, ontology, embedder, token_pooling="none", cos_theta=0.83):
        self.ontology = ontology
        self.cos_theta = cos_theta
        self.embedder = embedder
        self.token_pooling = token_pooling


    def string_match(self, tokens, execute=True):
        matches, matched_strings = [], []
        if not execute:
            return matches, matched_strings
        tokens = [token.lower() for token in tokens]
        for entity_string, entity_values in self.ontology.entities.items():
            tokenized_string = [token.lower() for token in self.embedder.tokenize(entity_string)]
            glued_string, _, _ = glue_subtokens(tokenized_string)
            string_length = len(glued_string)
            type_ = entity_values["type"]
            
            for occ in KnuthMorrisPratt(tokens, glued_string): 
                match = (occ, occ+string_length, type_)
                matches.append(match)
                matched_strings.append(entity_string)
                
        return matches, matched_strings


    def vote(self, similarities, neighbors):
        voter_types, voter_strings, full_terms = [], [], []
        weight_counter = Counter()
        for similarity, neighbor in zip(similarities, neighbors):
            type_, string, full_term = self.ontology.fetch_entity(neighbor)
            weight_counter[type_] += similarity
            voter_types.append(type_)
            voter_strings.append(string)
            full_terms.append(full_term)
        
        voted_type = weight_counter.most_common(1)[0][0]

        return voted_type, voter_types, voter_strings, full_terms


    def embedding_match(self, sentence_embeddings, sentence_subtokens, glued2tok, glued_tokens, execute=True):
        matches = []
        if not execute:
            return matches

        # Get embeddings of noun phrase chunks
        nps, nps_spans = noun_phrases(glued_tokens)
        nps_embeddings = []
        token2np, np2token = [], []
        all_tokens = []
        for i, (np_start, np_end) in enumerate(nps_spans):
            np_embeddings, matched_tokens = self.embedder.reduce_embeddings(sentence_embeddings, 
                np_start, np_end, sentence_subtokens, glued2tok, self.token_pooling)
            np2token.append(len(token2np))
            all_tokens += matched_tokens
            for emb in np_embeddings:
                nps_embeddings.append(emb.numpy())
                token2np.append(i)

        if not nps_embeddings:
            return matches

        # Classify noun chunks based on similarity threshold with nearest ontology concept
        q = np.stack(nps_embeddings)
        q_norm = preprocessing.normalize(q, axis=1, norm="l2")
        S, I = self.ontology.entity_index.search(q_norm, 1)
        S, I = S.reshape(len(S)), I.reshape(len(S))

        for i, (np_start, np_end) in enumerate(nps_spans):
            np_slice = np2token[i:i+2]
            if len(np_slice)==1: # last of spans
                np_slice.append(np_slice[-1]+1)
            start, end = np_slice[0], np_slice[-1]
            similarities = S[start:end]
            neighbors = I[start:end]
            tokens = all_tokens[start:end]
            type_, _, _, _ = self.vote(similarities, neighbors)
            confidence = similarities.mean()
            if confidence > self.cos_theta:
                # print(nps[i], type_, confidence)
                matches.append((np_start, np_end, type_))  

        return matches


    def combined_match(self, string_matches, embedding_matches, execute=True):
        matches = []
        if not execute:
            return matches
        matches = set(string_matches+embedding_matches)

        return matches


class RelationMatcher:
    def __init__(self, ontology):
        self.ontology = ontology

    def add_pattern(self, pattern):
        self.ontology.patterns.append(pattern)

    def pattern_match(self, tokens, entities, entity_symbol="<ENT>", verbose=False):
        query = copy.deepcopy(tokens)
        prev_end = 0
        regex_string = ""
        char2entity= []
        for i, entity in enumerate(entities):
            preceding_context_and_entity = " ".join(tokens[prev_end:entity["start"]]+[entity_symbol])
            regex_string += (preceding_context_and_entity + " ")
            char2entity += [i for c in (preceding_context_and_entity + " ")]
            prev_end = entity["end"]
        
        # retrieve relations
        matches = []
        for pattern in self.ontology.patterns:
            matches += pattern.match(regex_string, char2entity)
        matches = list(set(matches)) # remove duplicates
        relations = [{"head":m[0], "tail":m[1], "type":m[2]} for m in matches]
        
        # check (temp)
        if relations and verbose:
            print(tokens)
            for entity in entities:
                print("\t ENT: {}".format(tokens[entity["start"]:entity["end"]]))
            print(regex_string)
            for r in relations:
                h_string = tokens[entities[r["head"]]["start"]:entities[r["head"]]["end"]]
                t_string = tokens[entities[r["tail"]]["start"]:entities[r["tail"]]["end"]]
                print("found relation: {} |{}| {}".format(h_string, r["type"], t_string))

        return relations

    def pair_match(self, entities):
        relations = []
        pairs = [(a, b) for a in range(0, len(entities)) for b in range(0, len(entities))]
        for head_index, tail_index in pairs:
            head = entities[head_index]["type"]
            tail = entities[tail_index]["type"]
            relation = self.ontology.relations.get(head, {}).get(tail, None)
            if relation:
                relations.append({"type": relation, "head": head_index, "tail": tail_index})
        
        return relations

class RelationPattern:
    def __init__(self, regex, type_, subject_position, subject):
        '''
        Args:
            regex (str): a regular expression where entities 
            type_ (str): name of the relation type resulting from the regex pattern
                            in the string are replaced by @entity_symbol (default = <ENT>) 
            subject_position (int): integer that indicates the index/position of the subject entity in
                            the 1-to-N (subject-to-N) relation pattern, with N objects
            subject (str): either head or tail to indicate whether the subject at @position 
                             is the head or the tail of the relation
        '''
        self.regex = regex
        self.type = type_
        self.subject_position = subject_position
        self.subject = "head" if subject=="head" else "tail"
        self.object = "head" if subject=="tail" else "tail"
        assert(self.subject!=self.object)

    def match(self, query, char2entity):
        matches = []
        result = re.search(self.regex, query)
        if result:  
            first_entity = char2entity[result.start()]
            last_entity = char2entity[result.end()-1]
            entities = [i for i in range(first_entity, last_entity+1)]
            subject_position = entities.pop(self.subject_position)
            for object_ in entities:
                match = {self.subject:subject_position, self.object:object_, "type":self.type}
                matches.append((match["head"], match["tail"], match["type"]))

        return matches



if __name__ == "__main__":
    pass
    # test = ["Multilinear", "sparse", "principal", "component", "analysis", "of", "sublinear", "rich", "data"]
    # test1 = ["We", "use", "an", "MLP", "with", "5", "hidden", "layers"]
    # test2 = ["ALBERT",  "achieves", "a", "92.28", "F1-score", "on", "the", "Stanford", "Question", "Answering", "Dataset"]
    # test2  = ["Throughout", "the", "nonparametric", "and", "parametric", "theory", ",", "we", "rely", "on", "several", "simple", "yet", "powerful", "oracle", "inequalities", "for", "GAN", ",", "which", "could", "be", "of", "independent", "interest"]
    # print(noun_phrases(test1)[0])
    # print(noun_phrases(test2)[0])


