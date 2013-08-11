from text.blob import TextBlob
from text.tokenizers import WordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import nltk
import math
import copy
import re


class WordGrapher(object):

    doc = ""
    blob = None
    docs = []

    bigrams = None
    trigrams = None

    tokens = None
    tokenizer = None

    stopwords = []
    try:
        stopwords.extend(nltk.corpus.stopwords.words('indonesian'))
        stopwords.extend(nltk.corpus.stopwords.words('english'))
    except IOError:
        pass

    def __init__(self, doc, docs=None):
        self.doc = doc.lower()

        self.tokenizer = StopwordsTokenizer(stopwords=self.stopwords)
        self.blob = TextBlob(text=self.doc, tokenizer=self.tokenizer)
        self.tokens = copy.deepcopy(self.blob.tokens)

        self.bigrams = self.bigramify(self.blob.tokens)
        self.tokens.extend(self.bigrams)

        self.trigrams = self.trigramify(self.blob.tokens)
        self.tokens.extend(self.trigrams)

        self.docs = docs

    def bigramify(self, tokens):
        return ["%s %s" % (words[0], words[1]) for words in bigrams(tokens)]

    def trigramify(self, tokens):
        return ["%s %s %s" % (words[0], words[1], words[2]) for words in trigrams(tokens)]

    def ngrams(self, n):
        return self.blob.ngrams(n=n)

    def freq(self, word, docs=None):
        if docs is None:
            return self.tokens.count(word)
        else:
            blob = TextBlob(text=docs, tokenizer=self.tokenizer)
            blob.tokens.extend(self.bigramify(blob))
            blob.tokens.extend(self.trigramify(blob))
            return blob.tokens.count(word)

    def tf(self, word):
        return self.freq(word=word) / float(self.doc_word_count())

    def doc_word_count(self):
        return len(self.tokens)

    def num_docs_containing(self, word):
        if self.docs is None:
            return 2
        else:
            count = 0
            for document in self.docs:
                if self.freq(word=word, docs=document) > 0:
                    count += 1
            return 1 + count

    def idf(self, word):
        if self.docs is None:
            docs_length = 1
        else:
            docs_length = len(self.docs)
        num_docs = self.num_docs_containing(word)
        return math.log(docs_length / float(num_docs))

    def tf_idf(self, word):
        return self.tf(word) * self.idf(word)

    def analyze(self, count=10, percentage=False):
        score = {
            'freq': {},
            'tf': {},
            'idf': {},
            'tf-idf': {},
            'tokens': {}
        }

        for token in self.tokens:
            score['freq'][token] = self.freq(token)
            score['tf'][token] = self.tf(token)
            score['tokens'] = self.tokens
            score['idf'][token] = self.idf(token)
            score['tf-idf'][token] = math.fabs(self.tf_idf(token))

        final = {}
        for token in score['tf-idf']:
            if token not in final:
                final[token] = score['tf-idf'][token]
            else:
                if score['tf-idf'][token] > final[token]:
                    final[token] = score['tf-idf'][token]

        if not percentage:
            return [item for item in sorted(final.items(), key=lambda x: x[1], reverse=True)[:count]]
        else:
            result = [item for item in sorted(final.items(), key=lambda x: x[1], reverse=True)[:count]]
            max = 0.0
            for item in result:
                if item[1] > max:
                    max = item[1]
            return [(item[0], "%.2f%%" % (item[1]/max*100)) for item in result]


class StopwordsTokenizer(WordTokenizer):

    stopwords = None
    regx = [
        '(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
        '(\@.*)?',
        '(\'.*)?'
    ]

    def __init__(self, stopwords):
        if stopwords is None:
            raise ValueError("Stopwords must not be None")

        if not isinstance(stopwords, list):
            raise TypeError("Stopwords must be a List")

        self.stopwords = stopwords
        super(StopwordsTokenizer, self).__init__()

    def tokenize(self, text, include_punc=True):
        tk = RegexpTokenizer("[\w']+", flags=re.UNICODE)
        for pattern in self.regx:
            text = re.sub(pattern=pattern, repl='', string=text)
        tokens = tk.tokenize(text=text)
        return [token for token in tokens if token not in self.stopwords]