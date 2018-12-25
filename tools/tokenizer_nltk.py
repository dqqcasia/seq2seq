#! /usr/bin/python
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import sys
import codecs

def tokenizer_nltk(f1,f2):
    for line in f1:
        tokens = nltk.word_tokenize(line.strip())
        print(' '.join(tokens),file=f2)


f1 = codecs.open(sys.argv[1],'r','utf-8')
f2 = codecs.open(sys.argv[2],'w','utf-8')

tokenizer_nltk(f1, f2)