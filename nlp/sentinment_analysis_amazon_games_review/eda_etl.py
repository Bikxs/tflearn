import pandas as pd
import os
from utils import *
import nltk
# We need to download several nltk artefacts to perform the preprocessing
nltk.download('averaged_perceptron_tagger', download_dir='nltk')
nltk.download('wordnet', download_dir='nltk')
nltk.download('stopwords', download_dir='nltk')
nltk.download('punkt', download_dir='nltk')
nltk.download('omw-1.4', download_dir='nltk')
nltk.data.path.append(os.path.abspath('nltk'))

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from tqdm import tqdm


def review():
    print("\nReview:")
    # Read the JSON file
    review_df = pd.read_json(os.path.join('data', 'Video_Games_5.json'), lines=True, orient='records')
    # Select on the columns we're interested in 
    review_df = review_df[["overall", "verified", "reviewTime", "reviewText"]]
    print(review_df.head())
    
    print("\nCleaning up:")
    print("Before cleaning up: {}".format(review_df.shape))
    review_df = review_df[~review_df["reviewText"].isna()]
    review_df = review_df[review_df["reviewText"].str.strip().str.len()>0]
    print("After cleaning up: {}".format(review_df.shape))
    
    print("\nChecking verified vs non-verified review count")
    verified_df = review_df.loc[review_df["verified"], :]
    print(verified_df["overall"].value_counts())
    
    print("\nMap rating to a positive/negative label")
    # Use pandas map function to map different star ratings to 0/1
    verified_df["label"]=verified_df["overall"].map({5:1, 4:1, 3:0, 2:0, 1:0})
    print(verified_df["label"].value_counts())
    
    print("\nShuffling the data")
    # We are sampling 100% of the data in a random fashion, leading to a shuffled dataset
    verified_df = verified_df.sample(frac=1.0, random_state=random_seed)

    # Splint the data to inputs (inputs) and targets (labels)
    inputs, labels = verified_df["reviewText"], verified_df["label"]

    
    return inputs, labels

def preprocess(inputs, labels):
    rerun = False

    # Define a lemmatizer (converts words to base form)
    lemmatizer = WordNetLemmatizer()

    # Define the English stopwords
    EN_STOPWORDS = set(stopwords.words('english')) - {'not', 'no'}

    # Code listing 9.2
    def clean_text(doc):
        """ A function that cleans a given document (i.e. a text string)"""
        
        # Turn to lower case
        doc = doc.lower()
        # the shortened form n't is expanded to not
        doc = re.sub(pattern=r"\w+n\'t ", repl="not ", string=doc)
        # shortened forms like 'll 're 'd 've are removed as they don't add much value to this task
        doc = re.sub(r"(?:\'ll |\'re |\'d |\'ve )", " ", doc)
        # numbers are removed
        doc = re.sub(r"/d+","", doc)
        # break the text in to tokens (or words), while doing that ignore stopwords from the result
        # stopwords again do not add any value to the task
        tokens = [w for w in word_tokenize(doc) if w not in EN_STOPWORDS and w not in string.punctuation]  
        
        # Here we lemmatize the words in the tokens
        # to lemmatize, we get the pos tag of each token and 
        # if it is N (noun) or V (verb) we lemmatize, else 
        # keep the original form
        pos_tags = nltk.pos_tag(tokens)
        clean_text = [
            lemmatizer.lemmatize(w, pos=p[0].lower()) \
            if p[0]=='N' or p[0]=='V' else w \
            for (w, p) in pos_tags
        ]

        # return the clean text
        return clean_text

    

    if rerun or \
        not os.path.exists(os.path.join('data','sentiment_inputs.pkl')) or \
        not os.path.exists(os.path.join('data','sentiment_labels.pkl')):
        # Apply the transformation to the full text
        # this is time consuming
        print("\nProcessing all the review data ... This can take a long time (~ 1hr)")
        tqdm.pandas()
        
        inputs = inputs.progress_apply(lambda x: clean_text(x))
        print("\tDone")
        
        print("Saving the data")
        inputs.to_pickle(os.path.join('data','sentiment_inputs.pkl'))
        labels.to_pickle(os.path.join('data','sentiment_labels.pkl'))
        
    else:
        # Load the data from the disk
        print("Data already found. If you want to rerun anyway, set rerun=True")
        inputs = pd.read_pickle(os.path.join('data', 'sentiment_inputs.pkl'))
        labels = pd.read_pickle(os.path.join('data', 'sentiment_labels.pkl'))
    return inputs, labels
def review_preprocessed(inputs_raw, inputs):
    for actual, clean in zip(inputs_raw.iloc[:5], inputs.iloc[:5]):
        print(f"Actual: {actual}\nClean: {clean}")
            
if __name__ == "__main__":
    inputs_raw, labels_raw = review()
    inputs, labels = preprocess(inputs_raw, labels_raw)
    review_preprocessed(inputs_raw,inputs)