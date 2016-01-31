import numpy as np
import scipy as sp
from scipy.special import gammaln
import h5py

importa lda_gibbs as ldag

if __name__ == '__main__':

    num_triples = '_10k'
    num_triples_val = '_1k'
    vName = 'training_utterances_tfidf'
    fn = {'topicModel': 'topicModel' + num_triples + '.npz',
          'raw_data_val': data_dir + '/' + 'Training_Shuffled_Dataset' + \
          num_triples_val +'.txt',
          'train_tfidf_h5': data_dir + '/' + 'training_utterances_tfidf' + \
          num_triples + '.h5',}

    h5f = h5py.File(fn['train_tfidf_h5'])
    m = h5f[vName][:]
    phi = np.load(fn['topicModel'])
    
    

