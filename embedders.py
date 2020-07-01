from abc import ABC, abstractmethod
import utils
import os, string
import torch
from transformers import BertModel, BertTokenizer

class Embedder(ABC):
    def __init__(self, embedding_size, indicator):
        self.embedding_size = embedding_size
        self.indicator = indicator
        super().__init__()

    
    def split(self, text):
        doc_tokens = []
        char_to_word_offset = []
        new_token = True
        for c in text:
            if utils.is_whitespace(c):
                new_token = True
            else:
                if c in string.punctuation:
                    doc_tokens.append(c)
                    new_token = True
                elif new_token:
                    doc_tokens.append(c)
                    new_token = False
                else:
                    doc_tokens[-1] += c
                    new_token = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        
        return doc_tokens, char_to_word_offset

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def embed(self):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

class BertEmbedder(Embedder):
    def __init__(self, pretrained_weights='bert-base-uncased', transformer_layer='last',
    embedding_size=768):
        
        self.pretrained_weights = pretrained_weights
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.encoder = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
        self.transformer_layer = transformer_layer
        self.layers = {'last':-1, 'penult':-2}

        super().__init__(embedding_size, transformer_layer+'_'+pretrained_weights)

    def tokenize(self, sequence):
        if isinstance(sequence, str):
            tokens = self.tokenizer.tokenize(sequence)
        else: # is list
            tokens = []
            for word in sequence:
                tokens += self.tokenizer.tokenize(word)

        return tokens

    def embed(self, sequence):
        indices = torch.tensor([self.tokenizer.encode(sequence, add_special_tokens=True)])
        with torch.no_grad():
            hidden_states = self.encoder(indices)[-1]
            embeddings = hidden_states[self.layers[self.transformer_layer]]
        
        return torch.squeeze(embeddings)
        
    def get_token_mapping(self, doc_tokens):
        ''' Returns mapping between BERT tokens
        and input tokens (what split_like_BERT gives).
        '''
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        return tok_to_orig_index, orig_to_tok_index


    def reduce_embeddings(self, embeddings, start, end, tokens, 
                          orig2tok, f_reduce="mean"):
        def _first(t, s):
            return t[0], [s[0]]

        def _mean(t, s):
            embedding = t.mean(dim=0)
            return [embedding], ["_".join(s)]

        def _max(t, s):
            embedding, _ = t.max(dim=0)
            return [embedding], ["_".join(s)]

        def _absmax(t, s):
            abs_max_indices = torch.abs(t).argmax(dim=0)
            embedding = t.gather(0, abs_max_indices.view(1,-1)).squeeze()  
            return [embedding], ["_".join(s)]

        def _none(t, s): 
            tokens = s if isinstance(s, list) else [s]
            return [emb for emb in t], tokens


        emb_positions = orig2tok[start:end+1]
        if len(emb_positions) == 1:  # last token in sentence
            emb_positions.append(emb_positions[-1]+1)
        emb_start, emb_end = emb_positions[0], emb_positions[-1]
        reduction = {"mean":_mean, "max": _max, "absmax":_absmax, 
            "first":_first, "none":_none}.get(f_reduce, _mean)
        selected_features = [emb.tolist() for emb in embeddings[emb_start:emb_end]]
        t = torch.FloatTensor(selected_features)
        embeddings, matched_tokens = reduction(t, tokens[emb_start:emb_end])  

        return embeddings, matched_tokens


    def __repr__(self):
        return "BertEmbedder()"

    def __str__(self):
        return "_BertEmbedder_{}Layer_{}Weights".format(self.transformer_layer, self.pretrained_weights)

def glue_subtokens(subtokens, seperators=["##"]):
    glued_tokens = []
    tok2glued = []
    glued2tok = []
    for i, token in enumerate(subtokens):
        for sep in seperators:
            if token.startswith(sep):
                glued_tokens[len(glued_tokens) - 1] = glued_tokens[len(glued_tokens) - 1] + token.replace(sep, '')
            else:
                glued2tok.append(i)
                glued_tokens.append(token)

            tok2glued.append(len(glued_tokens) - 1)

    return glued_tokens, tok2glued, glued2tok