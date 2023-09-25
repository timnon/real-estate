import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os
import sys
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import PreTrainedTokenizerFast

# To have a progress bar when running .apply
tqdm.pandas()

DATA_DIR = '../data'
REVIEWS_FILE = f'{DATA_DIR}/recommendations/reviews.csv'
REVIEWS_SCORED_FILE = f'{DATA_DIR}/recommendations/reviews_scored_10000.csv'

TOKENIZER = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
MODEL = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def get_sent_bert(text: str) -> int:
    # Get hidding state as text embedding
    token = TOKENIZER.encode(text, return_tensors = 'pt', truncation=True, max_length=512)
    result = MODEL(token)
    stars = int(torch.argmax(result.logits))+1
    return stars


def create_sentiment_sample(n=10_000):
    '''create reviews sample to run expensive model on'''
    df = pd.read_csv(REVIEWS_FILE)
    df = df.sample(n)
    df['score'] = df['comments'].fillna('').progress_apply(get_sent_bert)
    df.to_csv(REVIEWS_SCORED_FILE)


def train_classifier():
    '''train simple classifier on sample and then run on complete data set'''
    import random
    from sklearn.model_selection import train_test_split
    #from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    df = pd.read_csv(REVIEWS_SCORED_FILE)

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    X = df['comments'].astype(str).fillna('').apply(clean_text).str.lower().values
    y = df['score'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # Create a pipeline with a CountVectorizer and a Naive Bayes classifier
    model = make_pipeline(CountVectorizer(min_df=5,max_df=0.5), LogisticRegression(class_weight='balanced',max_iter=10_000))

    # Train the model
    model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    print('do inference')
    df_inf = pd.read_csv(REVIEWS_FILE)
    X = df_inf['comments'].astype(str).fillna('').apply(clean_text).str.lower().values
    df_inf['score'] = model.predict(X)
    df_inf.to_csv(REVIEWS_FILE+'.scored')




if __name__ == "__main__":
    #create_sentiment_sample()
    train_classifier()
