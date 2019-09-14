# NaiveBayes.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).


# Will McMurtry

import sys
import time

import numpy as np
from Eval import Eval

from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA=ALPHA
        
        #TODO: Initalize parameters
        self.vocab = X.shape[1]
        self.docs = X.shape[0]

        # Initialize logprior
        self.N_doc = len(Y)
        self.N_pos = np.count_nonzero(Y == 1.0)
        self.N_neg = self.N_doc - self.N_pos
        logprior = dict()
        logprior['pos'] = np.log(self.N_pos/self.N_doc)
        logprior['neg'] = np.log(self.N_neg/self.N_doc)
        self.logprior = logprior
        
        # Initialize log likelihoods
        self.log_likelihood_pos = np.zeros(self.vocab)
        self.log_likelihood_neg = np.zeros(self.vocab)

        ## build word counts for all pos and all neg articles    
        self.pos_corpus = np.zeros(self.vocab)
        self.neg_corpus = np.zeros(self.vocab)
        
        # Loop through each document
        for doc, label in enumerate(Y):

            # get nonzero indices of document  
            for word in X.getrow(doc).nonzero()[1]:

                # add counts of present words to pos or neg corpus
                if label == 1.0:
                    self.pos_corpus[word] += X[doc, word]
                else:
                    self.neg_corpus[word] += X[doc, word] 
                  
        self.Train(X,Y)

    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        pos_denominator = np.sum(self.pos_corpus) + self.ALPHA * self.vocab
        neg_denominator = np.sum(self.neg_corpus) + self.ALPHA * self.vocab

        # loop through each word in vocabulary
        for word in range(self.vocab):

            # Get log conditional probability for each class
            self.log_likelihood_pos[word] = np.log((self.pos_corpus[word] + self.ALPHA) / pos_denominator)
            self.log_likelihood_neg[word] = np.log((self.neg_corpus[word] + self.ALPHA) / neg_denominator)
            
        return

    def Predict(self, X):
        #TODO: Implement Naive Bayes Classification

        # Initialize result array
        predictions = np.zeros(self.docs)

        # Iterate through each document
        for doc in range(self.docs):

            # Initialize log likelihoods
            guess = [0,0]
            guess[0] = self.logprior['pos']
            guess[1] = self.logprior['neg']

            # Add probability of each word in document to guess array
            for word in X.getrow(doc).nonzero()[1]:
                guess[0] += self.log_likelihood_pos[word]

                guess[1] += self.log_likelihood_neg[word]
            #guess[0] = X.getrow(doc).dot(self.log_likelihood_pos)
            #guess[1] = X.getrow(doc).dot(self.log_likelihood_neg)
            
            
            # Inference
            prediction = np.argmax(guess)
            

            # add to result array
            if prediction == 0:
                predictions[doc] = 1.0
            else:
                predictions[doc] = -1.0
        
        return predictions

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

if __name__ == "__main__":
    start = time.time()
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    print(nb.Eval(test.X, test.Y))
    print(time.time() - start)
