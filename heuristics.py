from utils import KnuthMorrisPratt
from embedders import glue_subtokens
from nltk.translate.ribes_score import position_of_ngram
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def noun_phrases(tokens):
    grammar = r"""
    NALL: {<NN>*<NNS>*<NNP>*<NNPS>*}
    NC: {<JJ>*<NALL>+}
    NP: {<NC>+}  

    """

    cp = nltk.RegexpParser(grammar)
    pos = nltk.pos_tag(tokens)
    # print(pos)
    result = cp.parse(pos)
    noun_phrases = []
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        np = ''
        for x in subtree.leaves():
            np = np + ' ' + x[0]
        noun_phrases.append(np.strip())

    spans = []
    for np in noun_phrases:
        splitted_np = np.split()
        start = position_of_ngram(tuple(splitted_np), tokens)
        end = start+len(splitted_np)
        np_tokens = tokens[start:end]
        if _alphabetical_sequence(np_tokens, threshold=0.4): 
            spans.append((start, end)) # note: len(spans) != len(noun_phrases)

    return noun_phrases, spans

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


def embedding_match(sentence_embeddings, sentence_subtokens, glued2tok, glued_tokens, 
                    ontology, embedder, execute=True, threshold=100, f_reduce="none", k=5):
    matches = []
    if not execute:
        return matches

    # Get embeddings of noun phrase chunks
    nps, nps_spans = noun_phrases(glued_tokens)
    nps_embeddings = []
    for np_start, np_end in nps_spans:
        np_embedding, _ = embedder.reduce_embeddings(sentence_embeddings, 
            np_start, np_end, glued_tokens, glued2tok, f_reduce)
        nps_embeddings.append(np_embedding.numpy())

    if not nps_embeddings:
        return matches

    # Classify noun chunks based on threshold with nearest ontology concept
    D, I = ontology.entity_index.search(np.stack(nps_embeddings), 1)
    mask = np.where(D < threshold, 1, 0).reshape(len(D))
    types = [ontology.entity_table[index]["type"] for row in I for index in row]
    lookalikes = [ontology.entity_table[index]["string"] for row in I for index in row]
    distances = D.reshape(len(D))
    for (start, end), valid, type_, distance, lookalike in zip(nps_spans, mask, types, distances, lookalikes):
        # print(glued_tokens[start:end], type_, distance, lookalike)
        if valid:
            print(glued_tokens[start:end], type_, distance, lookalike)
            matches.append((start, end, type_))

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

