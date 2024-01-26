# Basic dependencies
import argparse
import logging
import os
import os.path as osp
import random
import shutil
import sys
import time
import itertools
import regex as re
import json
import ast
from collections import Counter

import numpy as np
import pandas as pd
import sklearn.metrics as evaluator

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as PS
from nltk.stem.snowball import SnowballStemmer as SS
from nltk.stem import WordNetLemmatizer as WNL

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import LinearSVC as SVM

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as GS

import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, fact_path, fake_path):
        self.fact_path = fact_path
        self.fake_path = fake_path

    def load_and_modify(self):
        with open(self.fact_path) as f_input:
            fact_data = [line.strip() for line in f_input]
        with open(self.fake_path) as f_input:
            fake_data = [line.strip() for line in f_input]
        fact_df = pd.DataFrame(fact_data, columns=['animal_description'])
        fact_df['label'] = 1
        fact_df['animal_type'] = fact_df['animal_description'].str.split().str[0]
        fact_df['animal_type'] = fact_df['animal_type'].str.replace("'", "")
        fact_df = fact_df[['animal_type', 'animal_description', 'label']]
        fact_df.to_csv('fact_df.csv', index=False)
        fake_df = pd.DataFrame(fake_data, columns=['animal_description'])
        fake_df['label'] = 0
        fake_df['animal_type'] = fake_df['animal_description'].str.split().str[0]
        fake_df['animal_type'] = fake_df['animal_type'].str.replace("'", "")
        fake_df = fake_df[['animal_type', 'animal_description', 'label']]
        fake_df.to_csv('fake_df.csv', index=False)
        return fact_df, fake_df


class Process:
    def __init__(self, fact_df, fake_df):
        self.fact_df = fact_df
        self.fake_df = fake_df

    def processing(self):
        record_accuracies = open('accuracies_pre_processing.txt', 'w')
        record_accuracies.write("[\n")
        list_of_strings = []
        combined_df = pd.concat([self.fact_df, self.fake_df], ignore_index=True)
        train_df, test_df = train_test_split(combined_df, test_size=0.20,
                                             stratify=combined_df[['animal_type', 'label']])
        train, test = train_test_split(train_df, test_size=0.20,
                                       stratify=train_df[['animal_type', 'label']])

        train.reset_index(inplace=True)
        test.reset_index(inplace=True)
        x_train = train.drop('label', axis=1)
        y_train = train['label']
        x_test = test.drop('label', axis=1)
        y_test = test['label']
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        stem_lemmatizer = ['Porter Stemmer', 'Snowball Stemmer', 'Lemmatizer']
        keep_lowercase = [True, False]
        remove_stop_words = [True, False]
        keep_punctuation = [True, False]

        for lower in keep_lowercase:
            for stop_words in remove_stop_words:
                for punctuation in keep_punctuation:
                    for process in stem_lemmatizer:
                        if process == 'Porter Stemmer':
                            ps = PS()
                        elif process == 'Snowball Stemmer':
                            ps = SS("english")
                        elif process == 'Lemmatizer':
                            ps = WNL()
                        if process == 'Porter Stemmer' or process == 'Snowball Stemmer':
                            corpus = []
                            for i in range(0, len(x_train)):
                                # remove punctuation
                                if punctuation:
                                    description = re.sub('[^a-zA-Z]', ' ', x_train['animal_description'][i])
                                # convert into lower
                                if lower:
                                    description = description.lower()
                                # split based on spaces
                                description = description.split()
                                # remove stop words
                                if stop_words:
                                    description = [ps.stem(word) for word in description
                                                   if not word in stopwords.words('english')]
                                else:
                                    description = [ps.stem(word) for word in description]
                                description = ' '.join(description)
                                corpus.append(description)

                            test_corpus = []
                            for i in range(0, len(x_test)):
                                # remove punctuation
                                if punctuation:
                                    description = re.sub('[^a-zA-Z]', ' ', x_test['animal_description'][i])
                                # convert into lower
                                if lower:
                                    description = description.lower()
                                # split based on spaces
                                description = description.split()
                                # remove stop words
                                if stop_words:
                                    description = [ps.stem(word) for word in description
                                                   if not word in stopwords.words('english')]
                                else:
                                    description = [ps.stem(word) for word in description]
                                description = ' '.join(description)
                                test_corpus.append(description)
                        if process == 'Lemmatizer':
                            corpus = []
                            for i in range(0, len(x_train)):
                                # remove punctuation
                                if punctuation:
                                    description = re.sub('[^a-zA-Z]', ' ', x_train['animal_description'][i])
                                # convert into lower
                                if lower:
                                    description = description.lower()
                                # split based on spaces
                                description = description.split()
                                # remove stop words
                                if stop_words:
                                    description = [ps.lemmatize(word) for word in description
                                                   if not word in stopwords.words('english')]
                                else:
                                    description = [ps.lemmatize(word) for word in description]
                                description = ' '.join(description)
                                corpus.append(description)

                            test_corpus = []
                            for i in range(0, len(x_test)):
                                # remove punctuation
                                if punctuation:
                                    description = re.sub('[^a-zA-Z]', ' ', x_test['animal_description'][i])
                                # convert into lower
                                if lower:
                                    description = description.lower()
                                # split based on spaces
                                description = description.split()
                                # remove stop words
                                if stop_words:
                                    description = [ps.lemmatize(word) for word in description
                                                   if not word in stopwords.words('english')]
                                else:
                                    description = [ps.lemmatize(word) for word in description]
                                description = ' '.join(description)
                                test_corpus.append(description)

                        vectorizer = ['Bag_of_words', 'Bag_of_words_tfidftransformer', 'tfidf_vectorizer']
                        n_gram = [(1, 1), (1, 2), (2, 2)]
                        for vector in vectorizer:
                            for ngram in n_gram:
                                if vector == 'Bag_of_words':
                                    cv = CountVectorizer(ngram_range=ngram)
                                    x_train_corpus = cv.fit_transform(corpus).toarray()
                                    x_test_corpus = cv.transform(test_corpus).toarray()
                                elif vector == 'Bag_of_words_tfidftransformer':
                                    vect_tfidf = Pipeline([('vect', CountVectorizer(ngram_range=ngram)),
                                                           ('tfidf', TfidfTransformer())])
                                    x_train_corpus = vect_tfidf.fit_transform(corpus).toarray()
                                    x_test_corpus = vect_tfidf.transform(test_corpus).toarray()
                                elif vector == 'tfidf_vectorizer':
                                    tfidf = TfidfVectorizer(ngram_range=ngram)
                                    x_train_corpus = tfidf.fit_transform(corpus).toarray()
                                    x_test_corpus = tfidf.transform(test_corpus).toarray()

                                model_list = ['BNB', 'MNB', 'LR', 'SVM']
                                for model in model_list:
                                    if model == 'BNB':
                                        classifier = BNB()
                                    elif model == 'MNB':
                                        classifier = MNB()
                                    elif model == 'LR':
                                        classifier = LR(random_state=42)
                                    elif model == 'SVM':
                                        classifier = SVM(random_state=42)

                                    classifier.fit(x_train_corpus, y_train)
                                    pred = classifier.predict(x_test_corpus)
                                    score = metrics.accuracy_score(y_test, pred)
                                    record_performance = '{"Lowercase": ' + str(lower).lower() \
                                                         + ', "Punctuation": ' + str(punctuation).lower() \
                                                         + ', "Stop words": ' + str(stop_words).lower() \
                                                         + ', "Stemmer/Lemmatization": "' + process \
                                                         + '", "vectorizer": "' + vector \
                                                         + '", "ngram_range": "' + str(ngram) \
                                                         + '", "Model": "' + model \
                                                         + '", "Accuracy": ' + str(score) + '},'
                                    record_accuracies.write(record_performance)
                                    record_accuracies.write("\n")
                                    list_of_strings.append(record_performance[:-1])
        record_accuracies.write("]")
        record_accuracies.close()
        # Initialize an empty list to store the dictionaries
        list_of_dicts = []

        # Convert strings to dictionaries
        for s in list_of_strings:
            dictionary = json.loads(s)
            list_of_dicts.append(dictionary)
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(list_of_dicts)
        # Group by 'model' and find the row with the maximum value of 'accuracy' in each group
        result = df.groupby('Model', as_index=False).apply(
            lambda group: group[group['Accuracy'] == group['Accuracy'].max()])
        # Keep only one row for each unique combination of column3 and column4
        final_result = result.drop_duplicates(subset=['Model', 'Accuracy'], keep='first')
        print(final_result)
        print(len(final_result))
        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        x_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        x_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        record_best_grid_parameters = open('best_grid_parameters_accuracies.txt', 'w')
        final_test_results = open('final_results.txt', 'w')
        for index, row in final_result.iterrows():
            if row['vectorizer'] == 'Bag_of_words':
                vectorizer = CountVectorizer
            if row['vectorizer'] == 'Bag_of_words_tfidftransformer' or row['vectorizer'] == 'tfidf_vectorizer':
                vectorizer = TfidfVectorizer
            if row['Stemmer/Lemmatization'] == "Porter Stemmer" or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                if row['Stemmer/Lemmatization'] == "Porter Stemmer":
                    ps = PS()
                if row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    ps = SS("english")

                class StemmedVectorizerPS(vectorizer):
                    def build_analyzer(self):
                        analyzer = super(StemmedVectorizerPS, self).build_analyzer()
                        return lambda doc: ([ps.stem(w) for w in analyzer(doc)])

            if row['Stemmer/Lemmatization'] == "Lemmatizer":
                wnl = WNL()

                class StemmedVectorizerLemma(vectorizer):
                    def build_analyzer(self):
                        analyzer = super(StemmedVectorizerLemma, self).build_analyzer()
                        return lambda doc: ([wnl.lemmatize(w) for w in analyzer(doc)])

            if not row['Stop words'] and row['Punctuation']:
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                            or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    stemmed_ps_vect = StemmedVectorizerPS(lowercase=row['Lowercase'],
                                                          ngram_range=ast.literal_eval(row['ngram_range']))

                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    stemmed_lemma_vect = StemmedVectorizerLemma(lowercase=row['Lowercase'],
                                                                ngram_range=ast.literal_eval(row['ngram_range']))
            if row['Stop words'] and not row['Punctuation']:
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                            or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    stemmed_ps_vect = StemmedVectorizerPS(lowercase=row['Lowercase'],
                                                          ngram_range=ast.literal_eval(row['ngram_range']),
                                                          stop_words='english',
                                                          token_pattern=None)

                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    stemmed_lemma_vect = StemmedVectorizerLemma(lowercase=row['Lowercase'],
                                                                ngram_range=ast.literal_eval(row['ngram_range']),
                                                                stop_words='english',
                                                                token_pattern=None)

            if row['Stop words'] and row['Punctuation']:
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                            or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    stemmed_ps_vect = StemmedVectorizerPS(lowercase=row['Lowercase'],
                                                          ngram_range=ast.literal_eval(row['ngram_range']),
                                                          stop_words='english')

                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    stemmed_lemma_vect = StemmedVectorizerLemma(lowercase=row['Lowercase'],
                                                                ngram_range=ast.literal_eval(row['ngram_range']),
                                                                stop_words='english')
            if row['Stop words'] and not row['Punctuation']:
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                            or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    stemmed_ps_vect = StemmedVectorizerPS(lowercase=row['Lowercase'],
                                                          ngram_range=ast.literal_eval(row['ngram_range']),
                                                          token_pattern=None)

                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    stemmed_lemma_vect = StemmedVectorizerLemma(lowercase=row['Lowercase'],
                                                                ngram_range=ast.literal_eval(row['ngram_range']),
                                                                token_pattern=None)

            if row["Model"] == "BNB":
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', BNB()),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', BNB()),
                                         ])
                parameters = {'clf__alpha': (1, 1e-1, 1e-2, 1e-3), }
                gs_clf = GS(text_clf, parameters, n_jobs=-1)
                gs_clf = gs_clf.fit(x_train['animal_description'], y_train)
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', BNB(alpha=gs_clf.best_params_['clf__alpha'])),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', BNB(alpha=gs_clf.best_params_['clf__alpha'])),
                                         ])
                text_clf = text_clf.fit(x_train['animal_description'], y_train)
                predicted = text_clf.predict(x_test['animal_description'])
                score = metrics.accuracy_score(y_test, predicted)
                final_test_results.write("Model parameters and accuracy after applying grid search on train and "
                                         "validation sets:")
                final_test_results.write("\n")
                final_test_results.write("{")
                for key, value in row.items():
                    final_test_results.write(str(key) + ":" + str(value) + ", ")
                final_test_results.write("}")
                final_test_results.write("\n")
                final_test_results.write("Accuracy: " + str(score))
                final_test_results.write("\n")
            elif row["Model"] == "MNB":
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', MNB()),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', MNB()),
                                         ])
                parameters = {'clf__alpha': (1, 1e-1, 1e-2, 1e-3), }
                gs_clf = GS(text_clf, parameters, n_jobs=-1)
                gs_clf = gs_clf.fit(x_train['animal_description'], y_train)
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', MNB(alpha=gs_clf.best_params_['clf__alpha'])),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', MNB(alpha=gs_clf.best_params_['clf__alpha'])),
                                         ])
                text_clf = text_clf.fit(x_train['animal_description'], y_train)
                predicted = text_clf.predict(x_test['animal_description'])
                score = metrics.accuracy_score(y_test, predicted)
                final_test_results.write("Model parameters and accuracy after applying grid search on train and "
                                         "validation sets:")
                final_test_results.write("\n")
                final_test_results.write("{")
                for key, value in row.items():
                    final_test_results.write(str(key) + ":" + str(value) + ", ")
                final_test_results.write("}")
                final_test_results.write("\n")
                final_test_results.write("Accuracy: " + str(score))
                final_test_results.write("\n")
            elif row["Model"] == "LR":
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', LR()),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', LR()),
                                         ])
                parameters = {'clf__C': (10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
                              'clf__max_iter': (100, 300),
                              'clf__penalty': ('l1', 'l2', 'elasticnet', None), }
                gs_clf = GS(text_clf, parameters, n_jobs=-1)
                gs_clf = gs_clf.fit(x_train['animal_description'], y_train)
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', LR(C=gs_clf.best_params_['clf__C'],
                                                    max_iter=gs_clf.best_params_['clf__max_iter'],
                                                    penalty=gs_clf.best_params_['clf__penalty'])),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', LR(C=gs_clf.best_params_['clf__C'],
                                                    max_iter=gs_clf.best_params_['clf__max_iter'],
                                                    penalty=gs_clf.best_params_['clf__penalty'])),
                                         ])
                text_clf = text_clf.fit(x_train['animal_description'], y_train)
                predicted = text_clf.predict(x_test['animal_description'])
                score = metrics.accuracy_score(y_test, predicted)
                final_test_results.write("Model parameters and accuracy after applying grid search on train and "
                                         "validation sets:")
                final_test_results.write("\n")
                final_test_results.write("{")
                for key, value in row.items():
                    final_test_results.write(str(key) + ":" + str(value) + ", ")
                final_test_results.write("}")
                final_test_results.write("\n")
                final_test_results.write("Accuracy: " + str(score))
                final_test_results.write("\n")
            elif row["Model"] == 'SVM':
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', SVM()),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', SVM()),
                                         ])
                parameters = {'clf__loss': ('hinge', 'squared_hinge'),
                              'clf__C': (10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01), }
                gs_clf = GS(text_clf, parameters, n_jobs=-1)
                gs_clf = gs_clf.fit(x_train['animal_description'], y_train)
                if row['Stemmer/Lemmatization'] == "Porter Stemmer" \
                        or row['Stemmer/Lemmatization'] == "Snowball Stemmer":
                    text_clf = Pipeline([('vect', stemmed_ps_vect),
                                         ('clf', SVM(C=gs_clf.best_params_['clf__C'],
                                                     loss=gs_clf.best_params_['clf__loss'])),
                                         ])
                if row['Stemmer/Lemmatization'] == "Lemmatizer":
                    text_clf = Pipeline([('vect', stemmed_lemma_vect),
                                         ('clf', SVM(C=gs_clf.best_params_['clf__C'],
                                                     loss=gs_clf.best_params_['clf__loss'])),
                                         ])
                text_clf = text_clf.fit(x_train['animal_description'], y_train)
                predicted = text_clf.predict(x_test['animal_description'])
                score = metrics.accuracy_score(y_test, predicted)
                final_test_results.write("Model parameters and accuracy after applying grid search on train and "
                                         "validation sets:")
                final_test_results.write("\n")
                final_test_results.write("{")
                for key, value in row.items():
                    final_test_results.write(str(key)+":"+str(value)+", ")
                final_test_results.write("}")
                final_test_results.write("\n")
                final_test_results.write("Accuracy on test set: " + str(score))
                final_test_results.write("\n")

            record_best_grid_parameters.write("Grid Search Accuracy: " + str(gs_clf.best_score_))
            record_best_grid_parameters.write("\n")
            record_best_grid_parameters.write("Grid Search Parameters: " + str(gs_clf.best_params_))
            record_best_grid_parameters.write("\n")
            record_best_grid_parameters.write("Best model configuration before grid search: {")
            for key, value in row.items():
                record_best_grid_parameters.write(str(key) + ":" + str(value) + ", ")
            record_best_grid_parameters.write("}")
            record_best_grid_parameters.write("\n")
        final_test_results.close()
        record_best_grid_parameters.close()


def main():
    fact_path = input("Please enter the fact file path:- ")
    fake_path = input("Please enter the fake file path:- ")
    pdl = DataLoader(fact_path, fake_path)
    fact_df, fake_df = pdl.load_and_modify()
    p = Process(fact_df, fake_df)
    p.processing()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info('Total time consumption: %ds', end_time - start_time)
