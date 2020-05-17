from itertools import groupby
from collections import defaultdict
import string
import json
import os
from pathlib import Path
import math
from tqdm import tqdm
import nltk
from nltk.util import ngrams
from nltk.translate.ribes_score import position_of_ngram


# Knuth-Morris-Pratt string matching
# David Eppstein, UC Irvine, 1 Mar 2002

def KnuthMorrisPratt(text, pattern):
    '''Yields all starting positions of copies of the pattern in the text.
    Calling conventions are similar to string.find, but its arguments can be
    lists or iterators, not just strings, it returns all matches, not just
    the first one, and it does not need the whole text in memory at once.
    Whenever it yields, it will have read the text exactly up to and including
    the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


def split_with_indices(s, c=' '):
    '''Split string and return start and end positions of words'''
    p = 0
    for k, g in groupby(s, lambda x:x==c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q
        p = q
    
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=="\xa0":
        return True
    return False

def is_file(path):
    if path == None:
        return False

    return os.path.isfile(path)    

def create_dir_structure(path_dict):
    for _, path in path_dict.items():
        directory = os.path.dirname(path)
        Path(directory).mkdir(parents=True, exist_ok=True)

def merge_list_dicts(d1, d2):
    """
    Merge two dicts with lists as default value,
    and removing duplicate values
    """
    for key, l2 in d2.items():
        l1 = d1.get(key, [])
        d1[key] = set(l1+l2)

    return d1

def create_spans(sequence):
    #TODO
    pass


def create_dir(path):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)


def index2char(tokens):
    char_tuples = []
    start = 0
    end = 0
    for token in tokens:
        end = start+len(token)
        char_tuples.append((start, end))
        start += len(token)+1

    return char_tuples  


def no_nones(l):
    no_nones = True
    for item in l:
        if item==None:
            no_nones = False
    
    return no_nones

def positions_of_ngram(ngram, sentence):
    positions = []
    for i, sublist in enumerate(ngrams(sentence, len(ngram))):
        if ngram == sublist:
            positions.append(i)

    return positions
    
if __name__ == '__main__':
    d1 = {}
    d2 = {}
    d1['banana'] = [1, 4, (6, 9)]
    d2['banana'] = [3, (6, 9), (4, 5)]
    d2['monkey'] = [2, 6]
    print(merge_list_dicts(d1, d2))

