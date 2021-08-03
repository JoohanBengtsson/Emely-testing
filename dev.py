import random

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


corpus_root = './'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
p_synonym = 1

for word in wordlists.words('test.txt'):
    synonymList = set()
    u = random.uniform(0,1)
    if wn.synsets(word) and u < p_synonym:
        syn_set =  wn.synsets(word)[0] # Can be several synonyms. Now we just extract the first one.
        synonyms = syn_set._lemma_names
        print("     " + random.choice(synonyms))
    else:
        print(word)