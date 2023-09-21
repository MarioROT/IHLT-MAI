import math

en_corpus = {"a":17000, "he":10000, "mail":3900, "sent":850, "to":25000, "mordorian":0}
cat_corpus = {"a":21000, "he":11900, "mail":420, "sent":910, "to":750, "mordorian":0}
corpus = {'en':en_corpus,'cat':cat_corpus} 
Ns = {'en':1300000, 'cat':1100000}
Bs = {'en':22600, 'cat':36800}

def MLE(sentence, lang = 'en'):
    totSum = 0
    for unigram in sentence.split(' '):
        totSum += corpus[lang][unigram]/Ns[lang]
    return math.log10(totSum)

def LID(sentence, lamb = 0.5, lang = 'en'):
    totSum = 0
    for unigram in sentence.split(' '):
        totSum += (corpus[lang][unigram]+lamb)/(Ns[lang]+lamb*Bs[lang])
    return math.log10(totSum)

sentences = ['he', 'he sent a', 'he sent a mail', 'he sent a mail to a mordorian']

def process_sentence(sentence):
    print('----> Sentence: ', sentence)
    print('-- MLE')
    print('Eng: ', MLE(sentence))
    print('Cat: ', MLE(sentence, lang = 'cat'))
    print('-- LID')
    print('Eng: ', LID(sentence))
    print('Cat: ', LID(sentence, lang = 'cat'))

for sentence in sentences:
    process_sentence(sentence)
