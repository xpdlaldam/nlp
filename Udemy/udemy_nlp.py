### 6/10/2024 13. Stemming & Lemmatization
## Stemming is very crude as it just chops off the end of the word => the result is not necessarily a real word ex) better -> better

## Lemmatization is more sophisticated as it uses actual rules of language => the true root word will be returned ex) better -> good

## Parts-of-Speech (POS) matters as "going" can be a noun but when the root form is "go" it is not a noun  

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("mice") # returns "mouse"

lemmatizer.lemmatize("going", pos=wordnet.NOUN) # returns "going"
lemmatizer.lemmatize("going", pos=wordnet.VERB) # returns 'go'

### 6/10/2024 12. Stopwords

