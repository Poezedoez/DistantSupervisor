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
    result = cp.parse(nltk.pos_tag(tokens))
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
        spans.append((start, end))

    return noun_phrases, spans


def proper_sentence(subtokens, symbol_threshold=0.2):
    ## Verb checking
    VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    glued_tokens, _, _ = glue_subtokens(subtokens)
    pos_tags = {t[1] for t in nltk.pos_tag(glued_tokens)}
    has_verbs = VERBS.intersection(pos_tags)

    ## Symbol checking
    alphabetical_sentence = False
    alphabet_tokens = len([s for s in glued_tokens if re.match("^[a-zA-Z]*$", s)])
    if float(alphabet_tokens)/float(len(glued_tokens)) > symbol_threshold:
        alphabetical_sentence = True
    proper = (has_verbs and alphabetical_sentence)
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
                    ontology, embedder, execute=True, threshold=0.80, f_reduce="mean"):
    matches = []
    if not execute:
        return matches

    # Get embeddings of noun phrase chunks
    nps, nps_spans = noun_phrases(glued_tokens)
    nps_embeddings = []
    for np_start, np_end in nps_spans:
        np_embedding, _ = embedder.reduce_embeddings(sentence_embeddings, 
                                                np_start, 
                                                np_end, 
                                                glued_tokens, 
                                                glued2tok, 
                                                f_reduce)
        nps_embeddings.append(np_embedding.numpy())

    if not nps_embeddings:
        return matches

    # Classify noun chunks based on threshold with ontology concepts
    nps_array = np.stack(nps_embeddings)
    similarities = cosine_similarity(nps_array, ontology.embedding_array)
    max_similarity_indices = similarities.argmax(axis=1)
    max_similarities = np.take(similarities, max_similarity_indices)
    for i, span in enumerate(nps_spans):
        if max_similarities[i] > threshold:
            entity_string, type_, count = ontology.fetch_entity(max_similarity_indices[i])
            # print(nps[i], "\t {:0.2f} \t ({}({}), {}), ".format(max_similarities[i], entity_string, count, type_))
            matches.append((span[0], span[1], type_))

    return matches


def combined_match(string_matches, embedding_matches, execute=True):
    matches = []
    if not execute:
        return matches
    matches = set(string_matches+embedding_matches)

    return matches

if __name__ == "__main__":
    test = ["Multilinear", "sparse", "principal", "component", "analysis", "of", "sublinear", "rich", "data"]
    print(noun_phrases(test)[0])