__author__ = 'Purav'

import os
import nltk
from nltk.tokenize import sent_tokenize
import random
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.util import ngrams

hotels = ['hotel','room','staff','hilton','james','monaco','sofitel','affinia','ambassador','hardrock','talbott','conrad','fairmont','hyatt','omni','homewood','knickerbocker','sheraton','swissotel','allegro','amalfi','intercontinental','palmer']

def writeReviews(rootdir,output):
    words = 0
    fo = open(output,"w")
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            with open(os.path.join(folder,filename),'r') as src:
                review = src.read()

                fo.write(review+"\n\n\n")
    fo.close()


def get_unigrams(review,polarity):
    features = {}
    features['polarity'] = polarity
    review = nltk.word_tokenize(review)
    unigrams = ngrams(review,1)
    for unigram in unigrams:
        if unigram in features.keys():
            features[unigram]+=1
        else:
            features[unigrams]=1
    return features

def get_trigrams(review,polarity):
    features = {}
    features['polarity'] = polarity
    review = nltk.word_tokenize(review)
    unigrams = ngrams(review,3)
    for unigram in unigrams:
        if unigram in features.keys():
            features[unigram]+=1
        else:
            features[unigrams]=1
    return features

def get_bigrams(review,polarity):
    features = {}
    features['polarity'] = polarity
    review = nltk.word_tokenize(review)
    bigrams = ngrams(review,2)
    trigrams = ngrams(review,3)
    unigrams = ngrams(review,1)
    for unigram in unigrams:
        if unigram in features.keys():
            features[unigram]+=1
        else:
            features[unigrams]=1
    for bigram in bigrams:
        if bigram in features.keys():
            features[bigram]+=1
        else:
            features[bigram]=1
    for trigram in trigrams:
        if trigram in features.keys():
            features[trigram]+=1
        else:
            features[trigram]=1
    return features


def getReviews(rootdir):
    reviews = []
    unique = []
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            with open(os.path.join(folder,filename),'r') as src:
                review = src.read()
                words = regexp_tokenize(review,"\w+")
                for word in words:
                    unique.append(word)
                reviews.append(review)
    return reviews

def countW(rootdir):
    reviews = []
    unique = []
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            with open(os.path.join(folder,filename),'r') as src:
                review = src.read()
                words = regexp_tokenize(review,"\w+")
                for word in words:
                    unique.append(word)
                reviews.append(review)
    unique = set(unique)
    uniqueR = []
    for w in unique:
        if w not in stopwords.words('english'):
            uniqueR.append(w)
    print (len(set(uniqueR)))

def get_sentimentFeatures(review,polarity):
    features = {}
    features['polarity'] = polarity
    words = nltk.word_tokenize(review)
    x = nltk.pos_tag(words)
    for word, pos in set(x):
        if pos == 'JJ' or pos == 'RB' or pos == 'NNS' or pos == 'NNP':
            features[word] = True
    return features

def calculateAGARI(rootdir):
    avgARI = 0
    count = 0
    uniqueWords = 0
    personalRatio = 0
    dollarCount = 0
    personalPronouns = ["i","me","we","our","ours","mine"]
    hotelName = 0
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            with open(os.path.join(folder, filename), 'r') as src:
                review = src.read()
                personal = 0
                sentences = sent_tokenize(review)
                s = len(sentences)
                capitals = 0
                words = regexp_tokenize(review,"\w+")
                for x in words:
                    if x in personalPronouns:
                        personal+=1
                    if x in hotels:
                        hotelName+=1
                w = len(words)
                unique = len(set(words))
                uniqueWords+=unique
                review = review.replace(" ","")
                flag = "f"
                for i in range(len(review)):
                    if review[i].isupper():
                        capitals+=1
                    if review[i] == '$':
                        flag = "t"
                if flag=="t":
                    dollarCount+=1
                c = len(review)
                ari =4.71*(float(c)/w)+0.5*(float(w)/s)-21.43
                avgARI += ari
                count += 1
                personalRatio += float(personal)/w
                #print(nltk.ne_chunk(review))
    print("\n"+rootdir)
    print("ARI : "+str(float(avgARI/count)))
    print("Unique words"+" "+str(uniqueWords/float(count)))
    print("Ratio personal : "+str(personalRatio/float(count)))
    print("DollarCount :"+str(dollarCount))

def get_features(review,polarity):
    features = {}
    uniqueWords = 0
    personalRatio = 0
    personal = 0
    misspelt = 0
    hotelName = 0
    personalPronouns = ["i","me","we","our","ours","mine"]
    sentences = sent_tokenize(review)
    sent = nltk.word_tokenize(review)

    s = len(sentences)
    wordsR = regexp_tokenize(review,"\w+")
    for x in wordsR:
        if x in personalPronouns:
            personal+=1
        #if x not in set(words.words()):
            #misspelt+=1
        if x in hotels:
            hotelName+=1
    w = len(wordsR)
    unique = len(set(wordsR))
    uniqueWords+=unique
    review = review.replace(" ","")
    c = len(review)
    cap = 0
    features['dollar'] = False
    for i in range(len(review)):
        if review[i].isupper:
            cap+=1
        if review[i] == '$':
            features['dollar'] = True
    ari =4.71*(float(c)/w)+0.5*(float(w)/s)-21.43
    capRatio = c/float(s)
    personalRatio += float(personal)/w
    features['uniqueWords'] = uniqueWords
    features['personalRatio'] = personalRatio
    features['ari'] = ari
    features['capRatio'] = capRatio
    features['polarity'] = polarity
    features['hotel'] = hotelName
    ngrams = get_bigrams(review,'x')
    sentiments = get_sentimentFeatures(review,'x')
    for x in ngrams.keys():
        features[x] = ngrams[x]
    for x in sentiments.keys():
        features[x] = sentiments[x]
    features['misspelt'] = misspelt
    return features

'''
calculateAGARI("negative_polarity/deceptive_from_MTurk")
calculateAGARI("negative_polarity/truthful_from_Web")
calculateAGARI("positive_polarity/deceptive_from_MTurk")
calculateAGARI("positive_polarity/truthful_from_TripAdvisor")
'''

pos_deceptive = getReviews("positive_polarity/deceptive_from_MTurk")
pos_truthful = getReviews("positive_polarity/truthful_from_TripAdvisor")
neg_deceptive = getReviews("negative_polarity/deceptive_from_MTurk")
neg_truthful = getReviews("negative_polarity/truthful_from_Web")

countW("root")

featureSets = []
posFeatureSet = []
negFeatureSet = []
count = 0
nGramFeatures= []
sentimentFeatures = []


count = 1
for review in pos_deceptive:
    print(count)
    count+=1
    nGramFeatures.append((get_bigrams(review,'positive'),'deceptive'))
    featureSets.append((get_features(review,'positive'),'deceptive'))
    posFeatureSet.append((get_features(review,'positive'),'deceptive'))
    sentimentFeatures.append((get_sentimentFeatures(review,'positive'),'deceptive'))
for review in pos_truthful:
    print(count)
    count+=1
    nGramFeatures.append((get_bigrams(review,'positive'),'truthful'))
    featureSets.append((get_features(review,'positive'),'truthful'))
    posFeatureSet.append((get_features(review,'positive'),'truthful'))
    sentimentFeatures.append((get_sentimentFeatures(review,'positive'),'truthful'))
for review in neg_deceptive:
    print(count)
    count+=1
    nGramFeatures.append((get_bigrams(review,'negative'),'deceptive'))
    negFeatureSet.append((get_features(review,'negative'),'deceptive'))
    featureSets.append((get_features(review,'negative'),'deceptive'))
    sentimentFeatures.append((get_sentimentFeatures(review,'negative'),'deceptive'))
for review in neg_truthful:
    print(count)
    count+=1
    nGramFeatures.append((get_bigrams(review,'negative'),'truthful'))
    featureSets.append((get_features(review,'negative'),'truthful'))
    negFeatureSet.append((get_features(review,'negative'),'truthful'))
    sentimentFeatures.append((get_sentimentFeatures(review,'negative'),'truthful'))
random.shuffle(featureSets)


#for review in neg_deceptive:
    #print(classifier.classify(get_features(review,'negative')))


writeReviews("positive_polarity/deceptive_from_MTurk","posDec.txt")
writeReviews("positive_polarity/truthful_from_TripAdvisor","postru.txt")
writeReviews("negative_polarity/deceptive_from_MTurk","negDec.txt")
writeReviews("negative_polarity/truthful_from_Web","negtru.txt")
random.shuffle(posFeatureSet)
random.shuffle(negFeatureSet)
foldsize = 160
accuracyG = 0
accuracyD = 0
accuracyNGram = 0
accuracySentiment = 0
for x in range(10):
    print(str(x)+" fold")
    sentiTestSet = sentimentFeatures[x*foldsize:(x+1)*foldsize]
    sentiTrainSet = sentimentFeatures[:(x-1)*foldsize]+sentimentFeatures[(x+1)*foldsize:]
    nTestSet = nGramFeatures[x*foldsize:(x+1)*foldsize]
    nTrainSet = nGramFeatures[:(x-1)*foldsize]+nGramFeatures[(x+1)*foldsize:]
    testset = featureSets[x*foldsize:(x+1)*foldsize]
    trainset = featureSets[:(x-1)*foldsize]+featureSets[(x+1)*foldsize:]
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    sentiClassifier = nltk.NaiveBayesClassifier.train(sentiTrainSet)
    nGramClassifier = nltk.NaiveBayesClassifier.train(nTrainSet)
    accuracyNGram += nltk.classify.accuracy(nGramClassifier,nTestSet)
    accuracyG += nltk.classify.accuracy(classifier,testset)
    dTree = nltk.DecisionTreeClassifier.train(trainset)
    accuracyD += nltk.classify.accuracy(dTree,testset)
    accuracySentiment+=nltk.classify.accuracy(sentiClassifier,sentiTrainSet)
foldsize = 80
accuracyP = 0
for x in range(10):
    posTestSet = posFeatureSet[x*foldsize:(x+1)*foldsize]
    posTrainSet = posFeatureSet[:(x-1)*foldsize]+posFeatureSet[(x+1)*foldsize:]
    pClassifier = nltk.NaiveBayesClassifier.train(posTrainSet)
    accuracyP += nltk.classify.accuracy(pClassifier,posTestSet)
accuracyN = 0
for x in range(10):
    negTestSet = posFeatureSet[x*foldsize:(x+1)*foldsize]
    negTrainSet = posFeatureSet[:(x-1)*foldsize]+posFeatureSet[(x+1)*foldsize:]
    nClassifier = nltk.NaiveBayesClassifier.train(negTrainSet)
    accuracyN += nltk.classify.accuracy(nClassifier,negTestSet)




print("Generic : "+str(accuracyG/float(10)))
print("N-Classifier : "+str(accuracyN/float(10)))
print("P-Classifier : "+str(accuracyP/float(10)))
print("Decision Trees: "+str(accuracyD/float(10)))
print("N Gram :"+str(accuracyNGram/float(10)))
print("Sentiment Classifier : "+str(accuracySentiment/float(10)))