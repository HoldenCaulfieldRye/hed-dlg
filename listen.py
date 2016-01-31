#!/usr/bin/env python
__docformat__ = 'restructedtext en'
__authors__ = ("Julian Serban, Alessandro Sordoni")
__contact__ = "Julian Serban <julianserban@gmail.com>"

import argparse
import cPickle
import traceback
import itertools
import logging
import time
import json
import sys
import search_v2 as search

import collections
import string
import os
import numpy
import codecs

import nltk
import theano
from random import randint
from simSearch import getBayesianSet
from dialog_encdec import DialogEncoderDecoder 
from numpy_compat import argpartition
from state import prototype_state
# sys.path.insert(0, '/disk1/data/hackathon/SkipToughts/skip-thoughts/')
# import skipthoughts
# 
logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def sample(model, seqs=[[]], n_samples=1, beam_search=None, ignore_unk=False,verbose=False): 
    if beam_search:
        sentences = [] 
         
        # seq = model.words_to_indices(seqs[0])
        gen_ids, gen_costs = beam_search.sample(seqs, n_samples, ignore_unk=ignore_unk,verbose=verbose) 
        
        return gen_ids
        # print len(gen_ids)
        
        # for i in range(len(gen_ids)):
        #     print gen_ids [i]
        #     sentence = model.indices_to_words(gen_ids[i])
        #     sentences.append(sentence)
        return sentences
    else:
        raise Exception("I don't know what to do")

# def rank_bs():

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")
       
    parser.add_argument("--ignore-unk",
            default=True, action="store_true",
            help="Ignore unknown words")
    
    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("--n-turns",
                        default=100, type=int,
                        help="Number of dialog turns to generate")

    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    state = prototype_state()
   
    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src)) 
    
    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
     
    model = DialogEncoderDecoder(state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    
    logger.info("This model uses " + model.decoder_bias_type + " bias type")

    beam_search = None
    sampler = None

    beam_search = search.RandomSampler(model)
    beam_search.compile()
    # print 'Loading sent2vec model'
    # model = skipthoughts.load_model()
    # print 'Done'
    sample_sentences = [ 
        'i love {0}',
        'what is your favourite {0} ?',
        'oh no , a {0} !',
        "what ' s the big deal about {0} ?",
        'i hate {0} ',
        'do you like {0} ?',
        "your friend ' s {0} looks so good",
        "i ' m sure you ' re not good at {0}",
        "i like my {0} strong",
        "where is the {0} ?"]
    sample_sentences = numpy.random.choice(sample_sentences,5)
    # Start chat loop    
    print "READY"
    sys.stdout.flush()
    while (True):
        all_samples = []
        word = raw_input("\n")
        utterances = []
        for sample_sentence in sample_sentences:
            utterances.append([sample_sentence.format(word)])
        context_samples, context_costs = beam_search.sample(utterances,n_samples = 1,n_turns=args.n_turns)
        all_samples += context_samples

        flat_samples = [item.replace("<s>",'').replace("</s>",'') for a in all_samples for item in a]
        flat_utterances = [item for a in utterances for item in a]
        # print 'Flat Samples:',flat_samples
        # print 'Flat Utterances:',flat_utterances 
        # query_vecs = skipthoughts.encode(model, flat_utterances)
        # search_vecs = skipthoughts.encode(model, flat_samples)

        # for i in range(len(utterances)):
        #     print 
        #     print utterances[i]
        #     print all_samples[i]
        # order, score = getBayesianSet(search_vecs, query_vecs, c=2)
        # print "Sorted samples: ",[flat_samples[i] for i in order]
        print json.dumps(all_samples)
        # sys.stdout.flush()


if __name__ == "__main__":
    # Run with THEANO_FLAGS=mode=FAST_RUN,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python chat.py Model_Name

    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    main()

