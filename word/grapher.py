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

    tfidf = None

    stopwords = []
    try:
        stopwords.extend(nltk.corpus.stopwords.words('indonesian'))
        stopwords.extend(nltk.corpus.stopwords.words('english'))
    except IOError:
        pass

    def __init__(self, doc=None, docs=None):
        self.tokenizer = StopwordsTokenizer(stopwords=self.stopwords)

        if doc:
            self.set_document(doc=doc)

        if docs:
            self.set_documents(docs=docs)

    def set_document(self, doc, docs_list_mode=False):
        if doc:
            return self.initialize_document(doc=doc, docs_list_mode=docs_list_mode)
        else:
            raise ValueError("Document must not be None or empty")

    def set_documents(self, docs):
        if docs and isinstance(docs, list) and len(docs) > 0:
            self.docs = [self.set_document(doc=doc, docs_list_mode=True) for doc in docs]
        else:
            raise ValueError("Documents must not be None or and empty List")

    def initialize_document(self, doc, docs_list_mode=False):
        if not docs_list_mode:
            self.doc = doc.lower()

            self.blob = TextBlob(text=self.doc, tokenizer=self.tokenizer)
            self.tokens = copy.deepcopy(self.blob.tokens)

            self.bigrams = self.bigramify(self.blob.tokens)
            self.tokens.extend(self.bigrams)

            self.trigrams = self.trigramify(self.blob.tokens)
            self.tokens.extend(self.trigrams)
        else:
            doc = doc.lower()

            blob = TextBlob(text=doc, tokenizer=self.tokenizer)
            tokens = copy.deepcopy(blob.tokens)

            bigram = self.bigramify(tokens=tokens)
            tokens.extend(bigram)

            trigram = self.trigramify(tokens=tokens)
            tokens.extend(trigram)

            return tokens

    def bigramify(self, tokens, as_string=True):
        if as_string:
            return ["%s %s" % (words[0], words[1]) for words in bigrams(tokens)]
        else:
            return bigrams(tokens)

    def trigramify(self, tokens, as_string=True):
        if as_string:
            return ["%s %s %s" % (words[0], words[1], words[2]) for words in trigrams(tokens)]
        else:
            return trigrams(tokens)

    def ngrams(self, n):
        return self.blob.ngrams(n=n)

    def freq(self, word, docs=None):
        if docs is None:
            return self.tokens.count(word)
        else:
            if not isinstance(docs, str):
                d = ""
                for item in docs:
                    d = "%s %s" % (d, item)
                docs = d

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
        if not self.doc or not self.docs:
            raise ValueError("Document and its Documents Set must not be None or empty")

        score = {
            'freq': {},
            'tf': {},
            'idf': {},
            'tf-idf': {}
        }

        for token in self.tokens:
            score['freq'][token] = self.freq(token)
            score['tf'][token] = self.tf(token)
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
            self.tfidf = [item for item in sorted(final.items(), key=lambda x: x[1], reverse=True)[:count]]
            return self.tfidf
        else:
            result = [item for item in sorted(final.items(), key=lambda x: x[1], reverse=True)[:count]]
            max = 0.0
            for item in result:
                if item[1] > max:
                    max = item[1]
            self.tfidf = [(item[0], "%.2f%%" % (item[1]/max*100)) for item in result]
            return self.tfidf

    def graph(self, word):
        return self.graph_doc(word=word)

    def graph_doc(self, word):
        if not self.tfidf:
            raise ValueError("Please call analyze first before creating a graph")

        result = {}
        tris = self.trigramify(tokens=self.blob.tokens, as_string=False)

        matches = ["%s %s %s" % (tri[0], tri[1], tri[2]) for tri in tris if word in tri[1]]
        result['tris'] = [item for item in self.tfidf if item[0] in matches]

        bis = self.bigramify(tokens=self.blob.tokens, as_string=False)
        matches = ["%s %s" % (bi[0], bi[1]) for bi in bis if word in bi[0] or word in bi[1]]
        result['twos'] = [item for item in self.tfidf if item[0] in matches]

        return result

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