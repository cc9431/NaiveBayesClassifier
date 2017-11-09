'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

Author: Charles Calder

################### NOTE ###################
# This runs on Python 2.7                  #
############################################

Date: 01.Nov.2017

'''
from __future__ import print_function
import sys
import math
import time
import random
import decimal

class NaiveBayes():
    '''
    Naive Bayes classifier for text data.
    Assumes input text is one text sample per line.  
    First word is classification, a string.
    Remainder of line is space-delimited text.
    '''
    
    def __init__(self,train):
        '''Create classifier using train, the name of an input training file.'''

        self.vocab = {}                 # nested dictionary(key = category, value = dictionary(key = word, value = word_count))
        self.uniques = {}               # dictionary of each word unique word
        self.classes_containing = {}    # dictionary(key = unique_word, value = num of classes that have that word)
        self.test_results = {}          # nested dictionary(key = category, value = dictionary(information about category))
        self.probType = ""              # reusable global variable for printing the current type of probability calculation
        self.word_count = 0             # total number of words in the document
        self.learn(train)               # loads train data, fills vocab and classes_containing dictionaries

    ####################################################################################################

    def learn(self,traindat):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''
        with open(traindat,'r') as fd:
            for line in fd.readlines():
                words = line.split()
                id = words.pop(0)                               # First word in every line is the category
                if not self.vocab.has_key(id):                  # If we havent seen the category before, add it the vocab
                    self.vocab[id] = {}
                    self.vocab[id]["num_words"] = 0
                self.vocab[id]["num_words"] += len(words)       # Track the number of words total to that category
                self.word_count += len(words)                   # Track the number of words total
                for word in words:
                    if self.vocab[id].has_key(word):            # Check if we have seen the word before in this category, if so, increment
                        self.vocab[id][word] += 1
                    else:
                        self.vocab[id][word] = 1                
                        if self.uniques.has_key(word):          # Check if we have seen this word ever
                            self.classes_containing[word] += 1  # If so, increment the number of categories containing this word
                        else:                                   
                            self.classes_containing[word] = 1   # Else, create new key for these two dictionaries
                            self.uniques[word] = "yes"

    def runTest(self, testdat, probType):
        '''Load the test data and run test on it with given probability type'''
        self.probType = probType    # Save the probability type for printing later
        self.test_results = {cat : {"Occurances" : 0, "Correct": 0, "NGuesses": 0} for cat in self.vocab} # Create dictionary for storing results
        with open(testdat,'r') as td:
            for line in td.readlines():
                category_probs = []             # This is where the list of 20 probabilities will be saved, each relating to a category
                words = line.split()
                real_id = words.pop(0)          # Save the real ID of the line to be checked against later
                self.test_results[real_id]["Occurances"] += 1
                for category in self.vocab:
                    cat_total = self.vocab[category]["num_words"]
                    priorProb = cat_total / float(self.word_count) # Calculate proir probability of category
                    running_prob = priorProb # Running probability is the product of all probabilities times the prior probability
                    for word in words:
                        # For m-estimate and tf-idf, use log space to solve underflow problem
                        if probType == "raw":
                            running_prob *= self.raw(category, word, cat_total)
                        elif probType == "mest":
                            running_prob += self.mest(category, word, cat_total)
                        else:
                            running_prob += self.tfidf(category, word, cat_total)
                    category_probs.append(running_prob) # Once all the probabilities have been gathered, add it to the list of probabilities
                guess_category = self.vocab.keys()[argmax(category_probs)]  # Find the highest of all probabilites
                self.test_results[guess_category]["NGuesses"] += 1          # Increment the number of guesses for the ID that was guessed
                if guess_category == real_id:                               # Check if the guess was correct
                    self.test_results[real_id]["Correct"] += 1

    ####################################################################################################

    def raw(self, category, word, cat_total):
        '''Calculate the raw probability of a category given a word'''
        if self.vocab[category].has_key(word):
            return self.vocab[category][word] / float(cat_total)
        return 0
                            
    def mest(self, category, word, cat_total):
        '''Calculate the m-estimate probability of a category given a word'''
        if self.vocab[category].has_key(word):
            return math.log((self.vocab[category][word] + 1) / float(cat_total + len(self.uniques)))
        return math.log(1 / float(cat_total + len(self.uniques)))

    def tfidf(self, category, word, cat_total):
        '''Calculate the tf-idf probability of a category given a word'''
        if self.classes_containing.has_key(word):
            idf = math.log((len(self.vocab) + 2) / float(self.classes_containing[word]))
        else:
            idf = math.log(len(self.vocab) + 2)
        if self.vocab[category].has_key(word):
            tf = (self.vocab[category][word] + 0.1) / float(cat_total)
        else:
            tf = 0.1 / float(cat_total + len(self.uniques))
        return math.log(tf * idf)

    ####################################################################################################

    def printTraining(self):
        '''Organized, formats, and prints data from the training set'''
        print("\n\n","############### TRAIN OUTPUT #########################")
        print("Num unique words: ", len(self.uniques))
        print("Word Count: ", self.word_count)
        print("{:^24}|{:^8}|{:^8}".format("Category", "NWords", "P(cat)"))
        for category in self.vocab:
            category_total = self.vocab[category]["num_words"]
            prob = category_total / float(self.word_count)
            print("{:^24}|{:^8}|{:^8.3f}".format(category, category_total, 100 * prob))

    def printTest(self):
        '''Organized, formats, and prints data from the testing set'''
        print("\n\n","############### TEST OUTPUT #########################")
        print("Probability Type: %s" % self.probType)
        print("{:^24}|{:^8}|{:^15}|{:^5}|{:^8}".format("Category", "NCorrect", "NGuesses", "N", "%Correct"))
        val = 0
        for category in self.test_results:
            guess = self.test_results[category]["NGuesses"]
            corr = self.test_results[category]["Correct"]
            occur = self.test_results[category]["Occurances"]
            val += 100 * corr / float(occur)
            print("{:^24}|{:^8}|{:^15}|{:^5}|{:^8.3f}".format(category, corr, guess, occur, 100 * corr / float(occur)))
        print("Average correct %",val/20)

    ####################################################################################################

def argmaxrandreturn(lst):
    '''
    If all the values in a list are zero:
        return a random index
    else:
        return the largest'''
    for val in lst:
        if val != 0:
            return lst.index(max(lst))
    return random.randint(0, len(lst) - 1)

def argmax(lst):
    '''Return the index of the largest value in a list'''
    return lst.index(max(lst))

    ####################################################################################################
    
def main():
    '''Take in a training set, learn it, and analyze a testing set with the given method'''
    t = time.time()
    if len(sys.argv) != 4:
        print("Usage: %s trainfile testfile probability-version" % sys.argv[0])
        sys.exit(-1)

    nbclassifier = NaiveBayes(sys.argv[1])
    nbclassifier.printTraining()
    if sys.argv[3] in ["raw", "mest", "tfidf"]:
        nbclassifier.runTest(sys.argv[2], sys.argv[3])
    else:
        print("Probability version not defined\nAvailable methods: raw, mest, tfidf")
        sys.exit(-1)

    nbclassifier.printTest()
    print ("Time: ", time.time() - t)

if __name__ == "__main__":
    main()
