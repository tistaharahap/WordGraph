from word.grapher import StopwordsTokenizer
import unittest


class TokenizerTest(unittest.TestCase):

    def test_tokenizer_with_stopwords_ok(self):
        stopwords = ['warga', 'kota']
        tokenizer = StopwordsTokenizer(stopwords=stopwords)
        result = tokenizer.tokenize(text="""Beruntungnya warga kota serba ada ini. Banyak hal yang bisa kita lakukan di 
        Jakarta, apalagi saat akhir pekan. Mari menikmati hari-hari libur pendek bersama, di kota kita!""")

        found = False
        for word in stopwords:
            if word in result:
                found = True

        self.assertIs(found, False, msg="Stopwords not filtered")
        self.assertIsNotNone(obj=result, msg="Tokenizer result must not be None")
        self.assertIsInstance(obj=result, cls=list, msg="Tokenizer result must be a List")

    def test_tokenizer_with_stopwords_not_list(self):
        stopwords = None
        tokenizer = None
        try:
            tokenizer = StopwordsTokenizer(stopwords=stopwords)
        except ValueError:
            pass
        except TypeError:
            pass

        self.assertIsNone(obj=tokenizer, msg="Tokenizer must be initialized with a list of Stopwords")