##### 09/21/2023
# token: a word
# span: words
# spacy is object oriented

import spacy

nlp = spacy.load("en_core_web_sm")

text1 = "Dr. Strange likes Peter. I love kimchi"
doc = nlp(text1)

for sentence in doc.sents:
    print(sentence)

for sentence in doc.sents:
    for word in sentence:
        print(word)

#### nltk
### not object oriented
### less user friendly but more customizable
### better for researchers

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
sent_tokenize(text1)
word_tokenize(text1)


##### 09/21/2023 E8
#### Tokenization: a process of splitting text into meaningful segments
nlp = spacy.blank('en')
type(nlp)

doc = nlp("Dr. Strange loves pav bhaji of mumbai as it costs only $2 per plate.")
doc = nlp('''"Let's go to N.Y.!"''')
type(doc)

for token in doc:
    print(token)

## span
span0 = doc[:4] # span
type(span0)
dir(span0)

## token
token0 = doc[0]
token1 = doc[1]

type(token0)
dir(token0) # methods
token0.is_alpha
token1.is_alpha

### Customize tokenization rule
doc = nlp("gimme double cheese extra large healthy pizza")
[token for token in doc]
tokens = [token.text for token in doc]
tokens

from spacy.symbols import ORTH
nlp.tokenizer.add_special_case("gimme", [
    {ORTH: "gim"},
    {ORTH: "me"},
    # {ORTH: "give"},
    # {ORTH: "me"},
])
doc = nlp("gimme double cheese extra large healthy pizza")
[token.text for token in doc]

### Sentence tokenization
doc = nlp("Dr. Strange loves pav bhaji of mumbai. Hulk loves chat of delhi")
[sentence in doc.sents] # error => need to add the 'sentencizer' component to the pipeline 

nlp.add_pipe('sentencizer')
nlp.pipe_names # ['sentencizer']
doc = nlp("Dr. Strange loves pav bhaji of mumbai. Hulk loves chat of delhi")
[sentence for sentence in doc.sents]


##### 09/24/2023 E9. NLP Pipelines
###
nlp = spacy.blank('en')
nlp.pipe_names # [] this means there are no pipelines defined. However loading things like spacy.load("en_core_web_sm") will have pipelines (downloaded in cmd: python -m spacy download en_core_web_sm)

###
nlp = spacy.load('en_core_web_sm')
nlp.pipe_names

# 'tok2vec', 
# 'tagger' - part of speech (pos)
# 'parser', 
# 'attribute_ruler', 
# 'lemmatizer', 
# 'ner' - named entity recognization

###
doc = nlp("Captain america ate $100 of samosa. Then he said I can do this all day.")
for token in doc:
    print(token, " | ", token.pos_, " | ", token.lemma_)

###
doc = nlp("Tesla Inc is going to acquire twitter for $40 billion")
for entity in doc.ents:
    print(entity.text, " | ", entity.label_, " | ", spacy.explain(entity.label_))

### Visualization
from spacy import displacy
displacy.render(doc, style="ent")

### If you want to use certain pipelines
nlp = spacy.blank("en")
source_nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("ner", source=source_nlp)
nlp.pipe_names ['ner']


##### 09/24/2023 E10. Stemming & Lemmatization
#### Stemming vs Lemmatization
### Stemming: using fixed rules such as removing able, ing etc to derive a base word
### Lemmatization: using knowledge of a language (aka linguistic knowledge) to derive a base word
import nltk
import spacy # does not support Lemmatization

## using nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

words = ["eating", "eats", "eat", "ate", "ability"]
for word in words:
    print(word, "|", stemmer.stem(word))

## using spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("eating eats eat ate ability")
for token in doc:
    print(token, "|", token.lemma_)

### Customization using attribute_ruler
nlp.pipe_names
ar = nlp.get_pipe('attribute_ruler')
ar.add([[{"TEXT": "bro"}], [{"TEXT": "brah"}], [{"TEXT": "bruh"}]], {"LEMMA": "brother"})
doc = nlp("bro brah bruh")
for token in doc:
    print(token.text, "|", token.lemma_)


##### 10/01/2023 E12. NER
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.pipe_names

doc = nlp("Tesla is going to acquire Twitter for $40 billion")
doc.ents
for entity in doc.ents:
    print(entity.text, " | ", entity.label_, " | ", spacy.explain(entity.label_))

nlp.pipe_labels
nlp.pipe_labels['ner']

### Customize NERs using set_ents: annotate tokens outside of any provided spans
from spacy.tokens import Span
s1 = Span(doc, 0, 1, label="ORG")
s2 = Span(doc, 5, 6, label="ORG")
doc.set_ents([s1, s2], default="unmodified")
for ent in doc.ents:
    print(ent.text, " | ", ent.label_)

## cosine similarity is very useful in nlp => look for this in codebasics youtube
## In NLP we call feature engineering "Text Representation"
## Representing text as a vector is called Vector Space Model

#### 11/06/2023
## Approaches of converting text into vector
# Label Encoding - mapping each word to numbers => inefficient => not used in NLP
# one-hot encoding => inefficient => not used in NLP
# Bag of Words
# TF-IDF
# Word Embeddings

#### 11/08/2023 Text representation using Bag of Words (BOW)
import pandas as pd
import numpy as np

df = pd.read_csv("spam.csv")
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=.2)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_cv = v.fit_transform(X_train.values)
X_train_cv.toarray()[:2]
X_train_cv.shape # (4457, 7722) 4_457 emails and each email has size = 7_722 (the number of word)

dir(v)
v.get_feature_names()
v.vocabulary_ # the ith index of each word
v.get_feature_names()[2871]

X_train_np = X_train_cv.toarray()
X_train_np[:4][0]
np.where(X_train_np[0] != 0)

X_train[:4][65]
X_train_np[0][232]
X_train_np[0][303]
X_train_np[0][830]
v.get_feature_names()[232]
v.get_feature_names()[303]
v.get_feature_names()[830]
v.get_feature_names()[926]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_cv, y_train)
X_test_cv = v.transform(X_test)

from sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)
print(classification_report(y_test, y_pred))

emails = [
    'up to 20% discount on parking',
    'up to 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
    'hey Peter can we hang out together to watch football',
]
emails_count = v.transform(emails)
model.predict(emails_count)

# Using Pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB()),
])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


#### 11/18/2023 Stop words: unimportant words ex) a, the, for
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

STOP_WORDS
len(STOP_WORDS) # 326

nlp = spacy.load("en_core_web_sm")
text = "We just opend our wings, the flying part is coming soon"
doc = nlp(text)
for token in doc:
    if token.is_stop:
        print(token)

# common nlp preprocessing: stemming & lemmatization, removing stop words
def preprocess(text):
    doc = nlp(text)
    
    # token.text makes the words strings: ''
    no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # no_stop_words = [token for token in doc if not token.is_stop and not token.is_punct]
    
    return no_stop_words

for token in doc:
    if token.is_stop:
        print(token.text)

preprocess("We just opend our wings, the flying part is coming soon")
preprocess("do you think it's going to work joanna we did have some discussions on moving and breaking up I don't know what we should do")
        
### from json use case
import pandas as pd

# lines=True: 1 lline per 1 json object
df = pd.read_json("combined.json", lines=True) 
df.shape
df.head(3)
df[df['topics'].str.len() != 0]

len(df['contents'].iloc[0])

df = df.head(10)

def preprocess2(text):
    doc = nlp(text)
    
    no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(no_stop_words)

## remove all stop words in the contents column
df['contents_no_stop_words'] = df['contents'].apply(preprocess2)


#### 11/19/2023 Bag of Words is just a simple count which ignores the order of the language, hence not always useful. Thus we use Bag of n-grams which uses pairs of words like a moving window. bi-gram uses two words as one pair, tri-gram uses three words as one pair. So Bag of Words is just a special case of Bag of n-grams, where n = 1

###
from sklearn.feature_extraction.text import CountVectorizer # ngram_range
v = CountVectorizer(ngram_range=(1,3))
text1 = "I am learning Spanish these days"
v.fit([text1]) # requires a dictionary
v.vocabulary_ # what do the numeric values mean? just indicies of the vector

###
import spacy

corpus = [
    "Thor ate pizza",
    "Loki is tall",
    "Thor is eating pizza",
]

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue # ignore
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

preprocess(corpus[0])
preprocess(corpus[1])
preprocess(corpus[2])

corpus_process = [preprocess(text) for text in corpus]
corpus_process

v = CountVectorizer(ngram_range=(1,2))
v.fit(corpus_process)
v.vocabulary_

## returns bag of n-grams
v.transform(["Thor eat pizza"]).toarray()
v.transform(["Hulk eat pizza"]).toarray()

import pandas as pd
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df.head(3)
df.shape

df['category'].value_counts().min()

# solve imbalance data: undersample
min_samples = df['category'].value_counts().min()

df_education = df[df['category'] == 'EDUCATION'].sample(min_samples, random_state = 42)
df_arts = df[df['category'] == 'CULTURE & ARTS'].sample(min_samples, random_state = 42)
df_latino = df[df['category'] == 'LATINO VOICES'].sample(min_samples, random_state = 42)

df_balanced = pd.concat([df_education, df_arts, df_latino])
df_balanced['category'].value_counts()

# convert str to numeric
target_map = {
    'EDUCATION': 0,
    'CULTURE & ARTS': 1,
    'LATINO VOICES': 2,
}
df_balanced['category_num'] = df_balanced['category'].map(target_map)

df_balanced2 = df_balanced[['short_description', 'category_num']]
df_balanced2

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced2['short_description'],
    df_balanced2['category_num'],
    test_size=.2,
    random_state=42,
    stratify=df_balanced2['category_num'] # makes as balanced as possible
)
X_train.shape
X_train.head(3)
y_train.head(3)
y_train.shape
y_train.value_counts()

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

clf_onegram = Pipeline([
     ('vectorizer_bow', CountVectorizer(ngram_range = (1, 1))),         
     ('Multi NB', MultinomialNB())         
])
clf_onegram.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

clf_twogram = Pipeline([
     ('vectorizer_bow', CountVectorizer(ngram_range = (1, 2))),
     ('Multi NB', MultinomialNB())         
])
clf_twogram.fit(X_train, y_train)
y_pred = clf_twogram.predict(X_test)
print(classification_report(y_test, y_pred))

## Using preprocessed text as X_train instead of raw text
df_balanced2
df_balanced2['preprocessed_txt'] = df_balanced2['short_description'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced2['preprocessed_txt'], # now using preprocessed text
    df_balanced2['category_num'],
    test_size=.2,
    random_state=42,
    stratify=df_balanced2['category_num']
)

clf_onegram = Pipeline([
     ('vectorizer_bow', CountVectorizer(ngram_range = (1, 1))),         
     ('Multi NB', MultinomialNB())         
])
clf_onegram.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


#### 11/26/2023 Text Representation Using TF-IDF
# Stopwords cannot solve every nlp problem. Although there may be words that are not of major focus, we also don't want to ignore them. In this case we count how many words appear from the entire document and we call this "Document Frequency (DF)" = number of times term t is present in all docs ex) If we have 4 documents, the word "that" appeared in 3 docs. We assign a lower score for high  DF because it could be a generic term, hence

# IDF(t) = total documents / number of docs term t is present (DF(t))

# We also want to consider word frequency but we also need to take into account "within" the same doc level. Think when there are 5000 words in doc1 but only 10 in doc2. We need to normalize them, hence 

# TF(t,d) total number of term t being present in doc1 / total number of tokens in doc1, where "TF = Term Frequency", t = term, and d = doc. 

# TF - IDF = TF(t,d) * IDF(t)

### Limitations of TF-IDF
## As n increases, dimensionality and sparsity increases
## Doesn't capture relationships among words
## Doesn't address out of vocab issue

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
    "Apple is announcing new iphone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating biryani and you are eating grapes"
]

v = TfidfVectorizer()
transformed_output = v.fit_transform(corpus)
v.vocabulary_ # the numbers are indices

dir(v)

all_feature_names = v.get_feature_names() # get_feature_names_out is deprecated

v.idf_

# note that words like amazon and apple are higher than is
for word in all_feature_names:
    idx = v.vocabulary_.get(word)
    print(f"{word} {v.idf_[idx]}")

corpus[:2]
transformed_output.toarray()[:2]


#### 12/03/2023 Text Representation Using TF-IDF (continued)
import pandas as pd

### Import
df = pd.read_csv("data/Ecommerce_data.csv")
df.shape # (24_000, 2)
df.head(5)
df['label'].unique()
df['label'].value_counts()

### Convert categories to numeric
df['label_num'] = df['label'].map({
    'Household' : 0, 
    'Books': 1, 
    'Electronics': 2, 
    'Clothing & Accessories': 3
})

### Packages
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

### W/o preprocessing
## train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=1,
    stratify=df['label_num']
)
y_train.value_counts()

## Model 1. KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
clf1 = Pipeline([
     ('vectorizer_tfidf', TfidfVectorizer()),    
     ('knn', KNeighborsClassifier())         
])

clf1.fit(X_train, y_train)

y_pred = clf1.predict(X_test)
print(classification_report(y_test, y_pred))

X_test[:3][8394]
X_test[:3][19213]
y_pred[:3]

## Model 2. MultinomialNB
from sklearn.naive_bayes import MultinomialNB
clf2 = Pipeline([
     ('vectorizer_tfidf', TfidfVectorizer()),    
     ('multi nb', MultinomialNB())         
])

clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_test)
print(classification_report(y_test, y_pred2))

X_test[:3][8394]
X_test[:3][19213]
y_pred2[:3]

## Model 3. RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf3 = Pipeline([
     ('vectorizer_tfidf', TfidfVectorizer()),    
     ('rf nb', RandomForestClassifier())         
])

clf3.fit(X_train, y_train)

y_pred3 = clf3.predict(X_test)
print(classification_report(y_test, y_pred3))

X_test[:3][8394]
X_test[:3][19213]
y_pred3[:3]

### W/ preprocessing
## Preprocessing function to remove stop words and lemmatization
import spacy
nlp = spacy.load("en_core_web_sm") 

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens) 

df['preprocessed_txt'] = df['Text'].apply(preprocess) # takes 10 mins
df

## train_test_split
X_train_prep, X_test_prep, y_train_prep, y_test_prep = train_test_split(
    df['preprocessed_txt'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=1,
    stratify=df['label_num']
)

## Model 3. RandomForestClassifier
clf3.fit(X_train_prep, y_train_prep)

y_pred3 = clf3.predict(X_test_prep)
print(classification_report(y_test_prep, y_pred3))

X_test_prep[:3][8394]
X_test_prep[:3][19213]
y_pred3[:3]



#### 12/04/2023 Word2Vec Part 2 | Implement word2vec in gensim | | Deep Learning Tutorial 42
import gensim
import pandas as pd

df = pd.read_json("data/Cell_Phones_and_Accessories_5.json", lines=True)
df

df['reviewText'][0]
gensim.utils.simple_preprocess(df['reviewText'][0])

review_text = df['reviewText'].apply(gensim.utils.simple_preprocess)
review_text

model = gensim.models.Word2Vec(
    window=10,
    min_count=2, # only use sentences having at least two or more words
    workers=4
)
model.build_vocab(review_text, progress_per=1_000)
model.epochs # 5
model.corpus_count # 194_439
model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)
model.save("saved_mod.model")
model.wv.most_similar("bad")
model.wv.similarity(w1="cheap", w2="inexpensive")
model.wv.similarity(w1="great", w2="awesome")
model.wv.similarity(w1="great", w2="shit")



#### 12/10/2023 Word vectors in Spacy overview
# cmd: !python -m spacy download en_core_web_lg
import spacy
nlp = spacy.load("en_core_web_lg")

doc1 = nlp("doc cat banana asdfasdf")
for token in doc1:
    print(token.text, token.has_vector, token.is_oov)

doc1[0].vector
doc1[0].vector.shape # (300,)

token1 = nlp("bread")
token1.vector.shape
token2 = nlp("sandwich burger car tiger human wheat")
for token in token2:
    print(f"{token.text} <-> {token1.text}:", token.similarity(token1))

for token in token2:
    print(f"{token1.text} <->{token.text}:", token1.similarity(token))

def fun(doc1, doc2):
    token1 = nlp(doc1)
    token2 = nlp(doc2)
    for token in token2:
        print(f"{token1.text} <-> {token.text}:", token1.similarity(token))
fun("bread", "sandwich burger")

def print_similarity(doc1, doc2):
    token1 = nlp(doc1)
    token2 = nlp(doc2)
    for token in token2:
        print(f"{token1.text} <-> {token.text}:", token1.similarity(token))
print_similarity("iphone", "samsung apple iphone dog")

# using cosine_similarity
king = nlp.vocab["king"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector
queen = nlp.vocab["queen"].vector
result = king - man + woman

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([result], [queen])