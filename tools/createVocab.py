"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Chunqi Wang in July 2017.
"""
import codecs
import os
import yaml
import logging
import sys
from argparse import ArgumentParser
from collections import Counter


# from utils import AttrDict


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


def make_vocab(fpath, fname):
    """Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        words = l.split()
        word2cnt.update(Counter(words))
    word2cnt.update({"<PAD>": 10000000000000,
                     "<UNK>": 1000000000000,
                     "<S>": 100000000000,
                     "</S>": 10000000000})
    with codecs.open(fname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            # fout.write(u"{}\t{}\n".format(word, cnt))
            fout.write(u"{}\n".format(word))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


make_vocab(sys.argv[1], sys.argv[2])

logging.info("Done")