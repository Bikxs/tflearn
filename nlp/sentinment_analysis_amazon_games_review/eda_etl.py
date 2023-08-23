import pandas as pd
import os
from utils import *
import nltk
from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer

NLTK_FOLDER = 'data/nltk'
N_VOCAB = 11888

def download_nltk():
    if not os.path.exists(NLTK_FOLDER):
        os.makedirs(NLTK_FOLDER)
    # We need to download several nltk artefacts to perform the preprocessing
    nltk.download('averaged_perceptron_tagger', download_dir=NLTK_FOLDER)
    nltk.download('wordnet', download_dir=NLTK_FOLDER)
    nltk.download('stopwords', download_dir=NLTK_FOLDER)
    nltk.download('punkt', download_dir=NLTK_FOLDER)
    nltk.download('omw-1.4', download_dir=NLTK_FOLDER)
    nltk.data.path.append(os.path.abspath(NLTK_FOLDER))


def review():
    # print("\nReview:")
    # Read the JSON file
    review_df = pd.read_json(os.path.join('data', 'Video_Games_5.json'), lines=True, orient='records')
    # Select on the columns we're interested in 
    review_df = review_df[["overall", "verified", "reviewTime", "reviewText"]]
    # print(review_df.head())

    # print("\nCleaning up:")
    # print("Before cleaning up: {}".format(review_df.shape))
    review_df = review_df[~review_df["reviewText"].isna()]
    review_df = review_df[review_df["reviewText"].str.strip().str.len() > 0]
    # print("After cleaning up: {}".format(review_df.shape))

    # print("\nChecking verified vs non-verified review count")
    verified_df = review_df.loc[review_df["verified"], :]
    # print(verified_df["overall"].value_counts())

    # print("\nMap rating to a positive/negative label")
    # Use pandas map function to map different star ratings to 0/1
    verified_df["label"] = verified_df["overall"].map({5: 1, 4: 1, 3: 0, 2: 0, 1: 0})
    # print(verified_df["label"].value_counts())

    # print("\nShuffling the data")
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
        doc = re.sub(r"(?:\'ll |\'re |\'d |\'ve |\'s )", " ", doc)
        # numbers are removed
        doc = re.sub(r"/d+", "", doc)
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
                if p[0] == 'N' or p[0] == 'V' else w \
            for (w, p) in pos_tags
        ]

        # return the clean text
        return clean_text

    if rerun or \
            not os.path.exists(os.path.join('data', 'sentiment_inputs.pkl')) or \
            not os.path.exists(os.path.join('data', 'sentiment_labels.pkl')):
        # Apply the transformation to the full text
        # this is time consuming
        print("\nProcessing all the review data ... This can take a long time (~ 1hr)")
        tqdm.pandas()

        inputs = inputs.progress_apply(lambda x: clean_text(x))
        print("\tDone")

        print("Saving the data")
        inputs.to_pickle(os.path.join('data', 'sentiment_inputs.pkl'))
        labels.to_pickle(os.path.join('data', 'sentiment_labels.pkl'))

    else:
        # Load the data from the disk
        print("Data already found. If you want to rerun anyway, set rerun=True")
        inputs = pd.read_pickle(os.path.join('data', 'sentiment_inputs.pkl'))
        labels = pd.read_pickle(os.path.join('data', 'sentiment_labels.pkl'))
    return inputs, labels


def review_preprocessed(inputs_raw, inputs, labels):
    for actual, clean, label in zip(inputs_raw.iloc[:5], inputs.iloc[:5], labels.iloc[:5]):
        print(f"{'-' * 100}\nActual: {actual}\nClean: {clean}\n Label{label}")


# Code listing 9.3
def train_valid_test_split(inputs, labels, train_fraction=0.8):
    """ Splits a given dataset into three sets; training, validation and test """

    # Separate indices of negative and positive data points
    neg_indices = pd.Series(labels.loc[(labels == 0)].index)
    pos_indices = pd.Series(labels.loc[(labels == 1)].index)

    n_valid = int(min([len(neg_indices), len(pos_indices)]) * ((1 - train_fraction) / 2.0))
    n_test = n_valid

    neg_test_inds = neg_indices.sample(n=n_test, random_state=random_seed)
    neg_valid_inds = neg_indices.loc[~neg_indices.isin(neg_test_inds)].sample(n=n_test, random_state=random_seed)
    neg_train_inds = neg_indices.loc[~neg_indices.isin(neg_test_inds.tolist() + neg_valid_inds.tolist())]

    pos_test_inds = pos_indices.sample(n=n_test, random_state=random_seed)
    pos_valid_inds = pos_indices.loc[~pos_indices.isin(pos_test_inds)].sample(n=n_test, random_state=random_seed)
    pos_train_inds = pos_indices.loc[
        ~pos_indices.isin(pos_test_inds.tolist() + pos_valid_inds.tolist())
    ]

    tr_x = inputs.loc[neg_train_inds.tolist() + pos_train_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    tr_y = labels.loc[neg_train_inds.tolist() + pos_train_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    v_x = inputs.loc[neg_valid_inds.tolist() + pos_valid_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    v_y = labels.loc[neg_valid_inds.tolist() + pos_valid_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    ts_x = inputs.loc[neg_test_inds.tolist() + pos_test_inds.tolist()].sample(frac=1.0, random_state=random_seed)
    ts_y = labels.loc[neg_test_inds.tolist() + pos_test_inds.tolist()].sample(frac=1.0, random_state=random_seed)

    return (tr_x, tr_y), (v_x, v_y), (ts_x, ts_y)


def word_frequency(tr_x):
    # Create a large list which contains all the words in all the reviews
    data_list = [w for doc in tr_x for w in doc]

    # Create a Counter object from that list
    # Counter returns a dictionary, where key is a word and the value is the frequency
    cnt = Counter(data_list)

    # Convert the result to a pd.Series 
    freq_df = pd.Series(list(cnt.values()), index=list(cnt.keys())).sort_values(ascending=False)
    # Print most common words
    print("Print most common words", freq_df.head(n=10))

    # Print summary statistics
    print("Frequency statistics", freq_df.describe())

    return freq_df


def sequence_length(tr_x):
    # Create a pd.Series, which contain the sequence length for each review
    seq_length_ser = tr_x.str.len()

    # Get the median as well as summary statistics of the sequence length
    print("\nSome summary statistics")
    print("Median length: {}\n".format(seq_length_ser.median()))
    print(seq_length_ser.describe())

    print("\nComputing the statistics between the 10% and 90% quantiles (to ignore outliers)")
    p_10 = seq_length_ser.quantile(0.1)
    p_90 = seq_length_ser.quantile(0.9)
    seq_length = seq_length_ser[(seq_length_ser >= p_10) & (seq_length_ser < p_90)]
    print(seq_length.describe(percentiles=[0.33, 0.66]))
    return seq_length


def make_tokenizer(tr_x, n_vocab):
    # Define a tokenizer that will convert words to IDs
    # words that are less frequent will be replaced by 'unk'
    tokenizer = Tokenizer(num_words=n_vocab, oov_token='unk', lower=False)

    # Fit the tokenizer on the data
    tokenizer.fit_on_texts(tr_x.tolist())

    return tokenizer


def check_tokenizer(tokenizer):
    # Checking the attributes of the tokenizer
    word = "game"
    wid = tokenizer.word_index[word]
    print("The word id for \"{}\" is: {}".format(word, wid))
    wid = 4
    word = tokenizer.index_word[wid]
    print("The word for id {} is: {}".format(wid, word))
    test_text = [
        ['work', 'perfectly', 'wii', 'gamecube', 'issue', 'compatibility', 'loss', 'memory'],
        ['loved', 'game', 'collectible', 'come', 'well', 'make', 'mask', 'big', 'almost', 'fit', 'face', 'impressive'],
        ["'s", 'okay', 'game', 'honest', 'bad', 'type', 'game', '--', "'s", 'difficult', 'always', 'die', 'depresses',
         'maybe', 'skill', 'would', 'enjoy', 'game'],
        ['excellent', 'product', 'describe'],
        ['level', 'detail', 'great', 'feel', 'love', 'car', 'game']
    ]

    test_seq = tokenizer.texts_to_sequences(test_text)

    for text, seq in zip(test_text, test_seq):
        print("Text: {}".format(text))
        print("Sequence: {}".format(seq))
        print("\n")


# Code listing 9.4
def get_tf_pipeline(text_seq, labels, batch_size=64, bucket_boundaries=[5, 15], max_length=50, shuffle=False):
    """ Define a data pipeline that converts sequences to batches of data """

    # Concatenate the label and the input sequence so that we don't mess up the order when we shuffle
    data_seq = [[b] + a for a, b in zip(text_seq, labels)]
    # Define the variable sequence dataset as a ragged tensor
    tf_data = tf.ragged.constant(data_seq)[:, :max_length]
    # Create a dataset out of the ragged tensor
    text_ds = tf.data.Dataset.from_tensor_slices(tf_data)

    text_ds = text_ds.filter(lambda x: tf.size(x) > 1)
    # Bucketing the data
    # Bucketing assign each sequence to a bucket depending on the length
    # If you define bucket boundaries as [5, 15], then you get buckets,
    # [0, 5], [5, 15], [15,inf]
    bucket_fn = tf.data.experimental.bucket_by_sequence_length(
        lambda x: tf.cast(tf.shape(x)[0], 'int32'),
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[batch_size, batch_size, batch_size],
        padded_shapes=None,
        padding_values=0,
        pad_to_bucket_boundary=False
    )

    # Apply bucketing
    text_ds = text_ds.map(lambda x: x).apply(bucket_fn)

    # Shuffle the data
    if shuffle:
        text_ds = text_ds.shuffle(buffer_size=10 * batch_size)

    # Split the data to inputs and labels
    text_ds = text_ds.map(lambda x: (x[:, 1:], x[:, 0]))

    return text_ds


def validate_bucketing():
    x = [[1, 2], [1], [1, 2, 3], [2, 3, 6, 4, 5], [2, 0, 9, 7], [2, 4, 214, 21], [3, 4, 42, 7, 3, 2, 45, 52],
         [3, 2, 6, 543, 2, 3243, 2, 134, 52, 23], [3, 32, 21, 3, 2, 4, 134, 45, 1, 1, 45]]
    y = [0, 0, 0, 1, 1, 1, 0, 0, 0]

    a = get_tf_pipeline(x, y, batch_size=2, bucket_boundaries=[3, 5], max_length=15, shuffle=True)

    for x, y in a.take(6):
        print('\n')
        print(x)
        print('\ty=', y)


def validate_pipeline(tr_x, tr_y, v_x, v_y):
    train_ds = get_tf_pipeline(tr_x, tr_y, shuffle=True)
    valid_ds = get_tf_pipeline(v_x, v_y)

    print("Some training data ...")
    for x, y in train_ds.take(2):
        print("Input sequence shape: {}".format(x.shape))
        print(y)

    print("\nSome validation data ...")
    for x, y in valid_ds.take(2):
        print("Input sequence shape: {}".format(x.shape))
        print(y)


def get_datasets(batch_size=64):
    download_nltk()
    inputs, labels = preprocess(None, None)
    # review_preprocessed(inputs_raw,inputs,labels)
    (tr_x, tr_y), (v_x, v_y), (ts_x, ts_y) = train_valid_test_split(inputs, labels)

    n_vocab =  N_VOCAB
    tokenizer = make_tokenizer(tr_x, n_vocab)
    tr_x = tokenizer.texts_to_sequences(tr_x.tolist())
    v_x = tokenizer.texts_to_sequences(v_x.tolist())
    ts_x = tokenizer.texts_to_sequences(ts_x.tolist())


    train_ds = get_tf_pipeline(tr_x, tr_y, batch_size=batch_size, shuffle=True)
    valid_ds = get_tf_pipeline(v_x, v_y, batch_size=batch_size)
    test_ds = get_tf_pipeline(ts_x, ts_y, batch_size=batch_size)

    return dict(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, tr_x=tr_x, tr_y=tr_y, v_x=v_x, v_y=v_y,
                ts_x=ts_x, ts_y=ts_y)


if __name__ == "__main__":
    download_nltk()
    inputs_raw, labels_raw = review()
    inputs, labels = preprocess(inputs_raw, labels_raw)
    # review_preprocessed(inputs_raw,inputs,labels)
    (tr_x, tr_y), (v_x, v_y), (ts_x, ts_y) = train_valid_test_split(inputs, labels)

    freq_df = word_frequency(tr_x)
    sequence_length(tr_x)

    n_vocab = (freq_df >= 25).sum()
    print("Using a vocabulary of size: {}".format(n_vocab))

    tokenizer = make_tokenizer(tr_x, n_vocab)
    check_tokenizer(tokenizer)
    # Convert all of train/validation/test data to sequences of IDs
    tr_x = tokenizer.texts_to_sequences(tr_x.tolist())
    v_x = tokenizer.texts_to_sequences(v_x.tolist())
    ts_x = tokenizer.texts_to_sequences(ts_x.tolist())

    validate_pipeline(tr_x, tr_y, v_x, v_y)
