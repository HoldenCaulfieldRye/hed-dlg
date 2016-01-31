"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""

import sys, os
import h5py, json
import numpy as np
import scipy as sp
import sklearn 
from scipy.special import gammaln
from sklearn.feature_extraction.text import TfidfVectorizer

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler(object):

    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics)
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        print "n_docs, vocab_size", n_docs, vocab_size
        self._initialize(matrix)

        for it in xrange(maxiter):
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z

            # FIXME: burn-in and lag!
            yield self.phi()

if __name__ == "__main__":
    import os
    import shutil

    N_TOPICS = 10
    DOCUMENT_LENGTH = 100
    FOLDER = "topicimg"

    def vertical_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a vertical bar.
        """
        m = np.zeros((width, width))
        m[:, topic_index] = int(document_length / width)
        return m.flatten()

    def horizontal_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a horizontal bar.
        """
        m = np.zeros((width, width))
        m[topic_index, :] = int(document_length / width)
        return m.flatten()

    def save_document_image(filename, doc, zoom=2):
        """
        Save document as an image.

        doc must be a square matrix
        """
        height, width = doc.shape
        zoom = np.ones((width*zoom, width*zoom))
        # imsave scales pixels between 0 and 255 automatically
        # sp.misc.imsave(filename, np.kron(doc, zoom))

    def gen_word_distribution(n_topics, document_length):
        """
        Generate a word distribution for each of the n_topics.
        """
        width = n_topics / 2
        vocab_size = width ** 2
        m = np.zeros((n_topics, vocab_size))

        for k in range(width):
            m[k,:] = vertical_topic(width, k, document_length)

        for k in range(width):
            m[k+width,:] = horizontal_topic(width, k, document_length)

        m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

        return m

    def gen_document(word_dist, n_topics, vocab_size, length=DOCUMENT_LENGTH,
                     alpha=0.1):
        """
        Generate a document:
            1) Sample topic proportions from the Dirichlet distribution.
            2) Sample a topic index from the Multinomial with the topic
               proportions from 1).
            3) Sample a word from the Multinomial corresponding to the topic
               index from 2).
            4) Go to 2) if need another word.
        """
        theta = np.random.mtrand.dirichlet([alpha] * n_topics)
        v = np.zeros(vocab_size)
        for n in range(length):
            z = sample_index(theta)
            w = sample_index(word_dist[z,:])
            v[w] += 1
        return v

    def gen_documents(word_dist, n_topics, vocab_size, n=500):
        """
        Generate a document-term matrix.
        """
        # How are docs represented? need to do the same with movietriples dataset
        # Is n input dimensionality? no, that's vocab_size because of bag of words
        # representation. n is size of dataset
        m = np.zeros((n, vocab_size))
        for i in xrange(n):
            # generates a doc. this is the bit where we need to parse the docs
            m[i, :] = gen_document(word_dist, n_topics, vocab_size)
        return m

    def getCorpus(dataset_fn):
        corpus = []
        print 'reading in data from %s...' % (dataset_fn.split('/')[-1])
        with open(dataset_fn, 'r') as f:
            corpus = f.readlines()
        corpus = [it.strip().split('\t') for it in corpus]
        corpus = [subit for it in corpus for subit in it]
        return corpus
    
    def get_documents(fn):
        """
        Get bag-of-word representation of text dataset, as a matrix.
        """
        vocab_size = None 
        vName = 'training_utterances_tfidf'
        if all(map(os.path.isfile, [fn['tfidf_h5'], fn['wordList']])):
            print "reading in vectorised data..."
            h5f = h5py.File(fn['tfidf_h5'], 'r')
            m = h5f[vName][:]            
            h5f.close()
            print "data read successfully"
        else:
            corpus = getCorpus(fn['raw_data'])
            vectorizer = TfidfVectorizer(min_df=1)
            print "vectorising data via tf-idf..."
            X = vectorizer.fit_transform(corpus)
            wordList = vectorizer.get_feature_names()
            m = X.toarray()
            # save_h5(data_dir, fn['tfidf_h5'], vName, m)
            print "saving vectorised data in hdf5 format..."
            h5f = h5py.File(fn['tfidf_h5'])
            h5f.create_dataset(vName, data=m)
            h5f.close()
            with open(fn['wordList'], 'w') as f:
                json.dump(wordList, f)

        print "n_docs, vocab_size", m.shape
        return m

    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.mkdir(FOLDER)

    # 10k has vocab size 15749
    num_triples = '_1k'
    data_dir = '/disk1/data/hackathon/MovieTriples'
    fn = {'raw_data': data_dir + '/' + 'Training_Shuffled_Dataset' + \
          num_triples +'.txt',
          'tfidf_h5': data_dir + '/' + 'training_utterances_tfidf' + \
          num_triples + '.h5',
          'wordList': data_dir + '/' + 'word_list' + num_triples + '.json',
          'topicModel': data_dir + '/' + 'topicModel' + num_triples + '.npz',
          'data_lda': data_dir + '/' + 'training_utterances_lda' + \
          num_triples + '.npz'}
    
    width = N_TOPICS / 2
    # vocab_size = width ** 2
    # word_dist = gen_word_distribution(N_TOPICS, DOCUMENT_LENGTH)
    # matrix = gen_documents(word_dist, N_TOPICS, vocab_size)
    matrix = get_documents(fn)
    sampler = LdaSampler(N_TOPICS)

    print "training..."
    for it, phi in enumerate(sampler.run(matrix)):
        print "Iteration", it
        print "Likelihood", sampler.loglikelihood()

    phi = sampler.phi()
    m_lda = np.dot(matrix, phi.transpose())
    corpus_np = np.asarray(getCorpus(fn['raw_data']))
    
    for topc in xrange(m_lda.shape[1]):
        print "top 10 utterances for topic %i:" % (topc)
        topc_score = m_lda[:,topc]
        top_idxs = np.argsort(topc_score)[::-1]
        print topc_score[top_idxs[:10]] # this works but get repeats
        corpus_np_s = corpus_np[top_idxs]
        print corpus_np_s[0]
        idx, count = 1, 1
        while count < 10 and idx < corpus_np_s.shape[0]:
            duplicate = False
            for i in xrange(idx):
                if corpus_np_s[i] == corpus_np_s[idx]:
                    duplicate = True
                    break
            if not duplicate:
                print corpus_np_s[idx]
                count +=1
            idx += 1
        

    np.save(fn['topicModel'], phi)
    np.save(fn['data_lda'], m_lda)
    print "saved topic model under", fn['topicModel']
    
        
        # if it % 5 == 0:
        #     for z in range(N_TOPICS):
        #         save_document_image("topicimg/topic%d-%d.png" % (it,z),
        #                             phi[z,:].reshape(width,-1))


