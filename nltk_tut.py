import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import wordnet
from nltk.classify import ClassifierI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union, brown
from nltk.tokenize import PunktSentenceTokenizer

import pickle
import io

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from statistics import mode
# nltk.download()
#tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
# print(sent_tokenize(EXAMPLE_TEXT))
# print(word_tokenize(EXAMPLE_TEXT))

#Stop words

stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(EXAMPLE_TEXT)
filtered_sentence = [w for w in word_tokens if not w in stop_words]

print(f'word tokens {word_tokens}')
print(f'filtered sentence {filtered_sentence}')


# PorterStemmer

# ps = PorterStemmer()
# example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
# for w in example_words:
#     print(ps.stem(w))


# new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
# words = word_tokenize(new_text)
#
# for w in words:
#     print(ps.stem(w))

#Tagging


# train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")


# train_text = brown.raw("cb27")
# sample_text = "The following link shows the user how to enter their PTO and look at their benefits."
# custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # print(tagged)
            # chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO|WR>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            # chunked.draw()
            print(chunked)

            #NameEntity
            # namedEnt = nltk.ne_chunk(tagged, binary=True)
            # namedEnt.draw()
            # print(namedEnt)

    except Exception as e:
        print(str(e))


# process_content()

#wordnet


#
# syns = wordnet.synsets('program')
#synset
# print(syns[4].name())
#just the word
# print(syns[4].lemmas()[0].name())
#definition
# print(syns[4].definition())
#examples
# print(syns[4].examples())



for w in filtered_sentence:
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(w):
        for l in syn.lemmas():
            # print(l)
            synonyms.append(l.name())
            if l.antonyms():
                # print(l.antonyms())
                antonyms.append(l.antonyms()[0].name())
    print(f'word {w}')
    print(f'synonyms {set(synonyms)}')
    print(f'antonyms {set(antonyms)}')

# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('boat.n.01')

# print(w1.wup_similarity(w2))

# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('car.n.01')

# print(w1.wup_similarity(w2))

# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('computer.n.01')

# print(w1.wup_similarity(w2))

#text classification
# import random
# from nltk.corpus import movie_reviews
#
# short_pos = io.open('positive.txt', encoding='latin-1').read()
# short_neg = io.open('negative.txt', encoding='latin-1').read()
# print(short_pos)
#
# documents = []
#
# for r in short_pos.split('\n'):
#     documents.append((r, 'pos'))
#
# for r in short_neg.split('\n'):
#     documents.append((r, 'neg'))


# commenting out documents to incorporate positive/negative txt
# documents = [(list(movie_reviews.words(fileid)), category)
#     for category in movie_reviews.categories()
#     for fileid in movie_reviews.fileids(category)]


# random.shuffle(documents)
# print(documents[1])

# all_words = []
# for w in movie_reviews.words():
#     all_words.append(w.lower())
# commenting out all_words to incorporate positive/negative txt


# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)
#
# for w in short_pos_words:
#     all_words.append(w.lower())
#
# for w in short_neg_words:
#     all_words.append(w.lower())
#
#
# all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
#
# print(all_words['stupid'])

# word_features = list(all_words.keys())[:5000]
# print(word_features)

# def find_features(document):
#     # words = set(document)
#     words = word_tokenize(document)
#     features = {}
#     for w in word_features:
#         features[w] = (w in words)
#     return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# featuresets = [(find_features(rev), category) for (rev, category) in documents]
# random.shuffle(featuresets)

# naive bayes

# training_sets = featuresets[:10000]
# testing_sets = featuresets[10000:]


# classifier = nltk.NaiveBayesClassifier.train(training_sets)

# save_classifier = open('naive_bays.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes /len(votes)
        return conf




# classifier_f = open('naive_bays.pickle', 'rb')
# classifier = pickle.load(classifier_f)
# classifier_f.close()
# print('Original Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(classifier, testing_sets)) * 100)
# classifier.show_most_informative_features(15)


# Scikit-Learn incorporation

# MNB_classifier = SklearnClassifier(MultinomialNB())

# save_classifier = open('MNB_classifier.pickle', 'wb')
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('MNB_classifier.pickle', 'rb')
# MNB_classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# MNB_classifier.train(training_sets)
# print('MNB Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(MNB_classifier, testing_sets)) * 100)
#
# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

# save_classifier = open('BernoulliNB_classifier.pickle', 'wb')
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('BernoulliNB_classifier.pickle', 'rb')
# BernoulliNB_classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# BernoulliNB_classifier.train(training_sets)
# print('BernoulliNB Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(BernoulliNB_classifier, testing_sets)) * 100)


#LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

# save_classifier = open('LogisticRegression_classifier.pickle', 'wb')
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('LogisticRegression_classifier.pickle', 'rb')
# LogisticRegression_classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# LogisticRegression_classifier.train(training_sets)
# print('LogisticRegression Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(LogisticRegression_classifier, testing_sets)) * 100)
#
# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

# save_classifier = open('SGDClassifier_classifier.pickle', 'wb')
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('SGDClassifier_classifier.pickle', 'rb')
# SGDClassifier_classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# SGDClassifier_classifier.train(training_sets)
# print('SGD Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(SGDClassifier_classifier, testing_sets)) * 100)

# SVCclassifier = SklearnClassifier(SVC())
# SVCclassifier.train(training_sets)
# print('SVC Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(SVCclassifier, testing_sets)) * 100)

# LinearSVC_classifier = SklearnClassifier(LinearSVC())

# save_classifier = open('LinearSVC_classifier.pickle', 'wb')
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('LinearSVC_classifier.pickle', 'rb')
# LinearSVC_classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# LinearSVC_classifier.train(training_sets)
# print('LinearSVC Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(LinearSVC_classifier, testing_sets)) * 100)
#
# NuSVC_classifier = SklearnClassifier(NuSVC())

# save_classifier = open('NuSVC_classifier.pickle', 'wb')
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('NuSVC_classifier.pickle', 'rb')
# NuSVC_classifier = pickle.load(classifier_f)
# classifier_f.close()
#
# NuSVC_classifier.train(training_sets)
# print('NuSVC Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(NuSVC_classifier, testing_sets)) * 100)
#
#
# voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
# print('Voted Naive Bayes Classifier algo percent :', (nltk.classify.accuracy(voted_classifier, testing_sets)) * 100)
#
# print('Classification: ', voted_classifier.classify(testing_sets[0][0]), ' conf: ', voted_classifier.confidence(testing_sets[0][0]))
#
# print("Classification:", voted_classifier.classify(testing_sets[0][0]), "Confidence %:",voted_classifier.confidence(testing_sets[0][0])*100)
# print("Classification:", voted_classifier.classify(testing_sets[1][0]), "Confidence %:",voted_classifier.confidence(testing_sets[1][0])*100)
# print("Classification:", voted_classifier.classify(testing_sets[2][0]), "Confidence %:",voted_classifier.confidence(testing_sets[2][0])*100)
# print("Classification:", voted_classifier.classify(testing_sets[3][0]), "Confidence %:",voted_classifier.confidence(testing_sets[3][0])*100)
# print("Classification:", voted_classifier.classify(testing_sets[4][0]), "Confidence %:",voted_classifier.confidence(testing_sets[4][0])*100)
# print("Classification:", voted_classifier.classify(testing_sets[5][0]), "Confidence %:",voted_classifier.confidence(testing_sets[5][0])*100)
