import pickle
import re
from nltk.corpus import stopwords
# from bs4 import BeautifulSoup

with open('Models/logreg.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# Load the CountVectorizer
with open('Models/count_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the TfidfTransformer
with open('Models/tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

sentraw=input("Please enter your tracnsaction text: ")
# sentraw = 'Transaction of AED 34.42 debited from your a/c ****0535 at adnoc ABU DHABI AE. Avl Bal is AED 17350.57'

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):

    # text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    date_patterns = [r'\d{2} \d{2} \d{2} \d{4}','date']
    currency_patterns = [
        r'aed \d+(\.\d{1,2})?',  
        r'aed\d+(\.\d{1,2})?',]
    for pattern in currency_patterns:
        text = re.sub(pattern, '', text)
    for pattern in date_patterns:
        text = re.sub(pattern, '', text)
    specific_texts = [
        r'avl card bal \d+trx',  # Matches "avl card bal 1458000trx"
        r'available account balance',  # Matches "available account balance"
        # Add more patterns as needed
    ]
    for pattern in specific_texts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text
sent=clean_text(sentraw)
# Preprocess the test data using the same CountVectorizer and TfidfTransformer
X_test_counts = vectorizer.transform([sent])
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Make predictions on the test data using the trained model
lrpred = loaded_model.predict(X_test_tfidf)

# Print the predicted label
print(sentraw)
print('Predicted Label:', lrpred[0])