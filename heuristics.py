from utils import KnuthMorrisPratt
from embedders import glue_subtokens
from nltk.translate.ribes_score import position_of_ngram
import nltk
from sklearn import preprocessing
import numpy as np
import re
from collections import Counter


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
        if _alphabetical_sequence(np_tokens, threshold=0.4): 
            selected_spans.append((start, end))
            selected_nps.append(np)

    return selected_nps, selected_spans

def _contains_verb(tokens):
    VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    pos_tags = {t[1] for t in nltk.pos_tag(tokens)}
    has_verbs = VERBS.intersection(pos_tags)

    return has_verbs

def _alphabetical_sequence(tokens, threshold=0.2):
    alphabetical_sequence = False
    alphabet_tokens = len([s for s in tokens if re.match("^[a-zA-Z]*$", s)])
    if float(alphabet_tokens)/float(len(tokens)) > threshold:
        alphabetical_sequence = True

    return alphabetical_sequence

def proper_sequence(tokens, symbol_threshold=0.2):
    has_verbs = _contains_verb(tokens)
    alphabetical_sequence = _alphabetical_sequence(tokens)
    proper = (has_verbs and alphabetical_sequence)
    # if not proper:
    #     v = "VERBS" if not has_verbs else ''
    #     a = "a-Z" if not alphabetical_sentence else ''
    #     print("The following sentence does not have {} {}".format(v, a))
    return proper


def string_match(tokens, ontology, embedder, execute=True):
    matches, matched_strings = [], []
    if not execute:
        return matches
    tokens = [token.lower() for token in tokens]
    for entity_string, entity_values in ontology.entities.items():
        tokenized_string = [token.lower() for token in embedder.tokenize(entity_string)]
        glued_string, _, _ = glue_subtokens(tokenized_string)
        string_length = len(glued_string)
        type_ = entity_values["type"]
        
        for occ in KnuthMorrisPratt(tokens, glued_string): 
            match = (occ, occ+string_length, type_)
            matches.append(match)
            matched_strings.append(entity_string)
            
    return matches, matched_strings


def vote(similarities, neighbors, ontology):
    voter_types, voter_strings, full_terms = [], [], []
    weight_counter = Counter()
    for similarity, neighbor in zip(similarities, neighbors):
        type_, string, full_term = ontology.fetch_entity(neighbor)
        weight_counter[type_] += similarity
        voter_types.append(type_)
        voter_strings.append(string)
        full_terms.append(full_term)
    
    voted_type = weight_counter.most_common(1)[0][0]

    return voted_type, voter_types, voter_strings, full_terms


def embedding_match(sentence_embeddings, sentence_subtokens, glued2tok, glued_tokens, 
                    ontology, embedder, execute=True, threshold=0.83, token_pooling="mean"):
    matches = []
    if not execute:
        return matches

    # Get embeddings of noun phrase chunks
    nps, nps_spans = noun_phrases(glued_tokens)
    nps_embeddings = []
    token2np, np2token = [], []
    all_tokens = []
    for i, (np_start, np_end) in enumerate(nps_spans):
        np_embeddings, matched_tokens = embedder.reduce_embeddings(sentence_embeddings, 
            np_start, np_end, sentence_subtokens, glued2tok, token_pooling)
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
    S, I = ontology.entity_index.search(q_norm, 1)
    S, I = S.reshape(len(S)), I.reshape(len(S))

    for i, (np_start, np_end) in enumerate(nps_spans):
        np_slice = np2token[i:i+2]
        if len(np_slice)==1: # last of spans
            np_slice.append(np_slice[-1]+1)
        start, end = np_slice[0], np_slice[-1]
        similarities = S[start:end]
        neighbors = I[start:end]
        tokens = all_tokens[start:end]
        type_, _, _, _ = vote(similarities, neighbors, ontology)
        confidence = similarities.mean()
        if confidence > threshold:
            # print(nps[i], type_, confidence)
            matches.append((np_start, np_end, type_))  

    return matches


def combined_match(string_matches, embedding_matches, execute=True):
    matches = []
    if not execute:
        return matches
    matches = set(string_matches+embedding_matches)

    return matches

if __name__ == "__main__":
    # test = ["Multilinear", "sparse", "principal", "component", "analysis", "of", "sublinear", "rich", "data"]
    test1 = ["We", "use", "an", "MLP", "with", "5", "hidden", "layers"]
    test2 = ["ALBERT",  "achieves", "a", "92.28", "F1-score", "on", "the", "Stanford", "Question", "Answering", "Dataset"]
    # test2  = ["Throughout", "the", "nonparametric", "and", "parametric", "theory", ",", "we", "rely", "on", "several", "simple", "yet", "powerful", "oracle", "inequalities", "for", "GAN", ",", "which", "could", "be", "of", "independent", "interest"]
    print(noun_phrases(test1)[0])
    print(noun_phrases(test2)[0])

