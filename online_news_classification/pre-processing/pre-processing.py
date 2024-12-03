import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from online_news_classification.functions import setup, manage_datasets
import logging
import time

import pandas as pd
import os
from dotenv import load_dotenv

nltk.download("stopwords")
nltk.download('wordnet')
load_dotenv()

def pre_processing_dataset(args):
    # Removal stop words
    stop_words = set(stopwords.words("english"))

    # Removal punctuation
    punc = string.punctuation

    # Lemmatization
    lm = WordNetLemmatizer()

    for chunk in pd.read_csv(args.input_file, sep=";", chunksize=1000):
        chunk['text'] = chunk['title'] + ' ' + chunk['abstract']
        chunk['tokenized_title'] = chunk['title'].apply(
            lambda title: " ".join([lm.lemmatize(str(word)) for word in word_tokenize(str(title)) if (lm.lemmatize(str(word)) not in stop_words and lm.lemmatize(str(word)) not in punc)]))
        chunk['tokenized_abstract'] = chunk['abstract'].apply(
            lambda abstract: " ".join([lm.lemmatize(str(word)) for word in word_tokenize(str(abstract)) if (lm.lemmatize(str(word)) not in stop_words and lm.lemmatize(str(word)) not in punc)]))
        chunk['tokenized_text'] = chunk['tokenized_title'] + ' ' + chunk['tokenized_abstract']
        chunk.to_csv(args.output_file, mode="a", index=True, index_label="Index", sep=";", header=not os.path.exists(args.output_file))

def main():
    args = setup.get_arg_parser_pre_processing().parse_args()
    start_time = setup.initialize("pre_processing_"+args.dataset)
    pre_processing_dataset(args)
    logging.info("--- %s seconds ---" % (time.time() - start_time))




if __name__ == "__main__":
    main()