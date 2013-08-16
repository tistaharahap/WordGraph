import word


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content

setup(
    name='wordgrapher',
    version=word.__version__,
    description='Word Graph utility built with NLTK and TextBlob',
    long_description=(read("README.md")),
    keywords='tf-idf nlp graph machine learning',
    license=read("LICENSE"),
    author='Batista Harahap',
    author_email='batista@bango29.com',
    url='https://github.com/tistaharahap/WordGraph',
    setup_requires=['nltk', 'textblob>=0.5.0', 'greenlet', 'gevent'],
    packages=['word', 'mmirman'],
    classifiers=(
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        "Topic :: Text Processing :: Linguistic",
    )
)