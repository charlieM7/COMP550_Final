from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re



def stop_words():
    not_stopwords = ['not', "don't", 'aren', 'don', 'ain', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'against', 'very',
                    "won't", 'wouldn', "wouldn't"]
    stopwords_list = set(stopwords.words('english'))
    for not_stopword in not_stopwords:
        stopwords_list.remove(not_stopword)
    return stopwords_list

def remove_stopwords(tokens, stopwords_list):
    cleaned_text = [word for word in tokens if word not in stopwords_list]
    return cleaned_text

def lemmatize(tokens, lemmatizer):
    lemma = [lemmatizer.lemmatize(token) for token in tokens]
    lemma = [lemmatizer.lemmatize(token, "v") for token in lemma]
    return ' '.join(lemma)

def preprocessing(datasets, stopwords_list, lemmatizer):

    for dataset in datasets:
        fpath = Path(f"data/cleaned_{dataset.name}.csv")
    #     if not fpath.exists():

        dataset['Text'] = dataset['Text'].apply(lambda x: x.lower())
        dataset['Text'] = dataset['Text'].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x))
        dataset['Text'] = dataset['Text'].apply(lambda x: re.sub(r"[^\w\s]", "", x))
        dataset['Text'] = dataset['Text'].apply(lambda x: re.sub(r'\d+', "", x))
        dataset['Text'] = dataset['Text'].apply(lambda x: re.sub(r"\s+", " ", x).strip())
        dataset['Text'] = dataset['Text'].apply(lambda x: word_tokenize(x))
        dataset['Text'] = dataset['Text'].apply(lambda x: remove_stopwords(x, stopwords_list))
        dataset['Text'] = dataset['Text'].apply(lambda x: lemmatize(x, lemmatizer))

        dataset.to_csv(fpath, index=False)

if __name__ == "__main__":
    # auto_labeled = pd.read_csv('data/auto_labeled_dataset.csv')
    # auto_labeled.name = 'auto_labeled'
    # hand_labeled = pd.read_csv('data/hand_labeled_dataset.csv')
    # hand_labeled.name = 'hand_labeled'
    # unlabeled = pd.read_csv('data/unlabeled_dataset.csv')
    # unlabeled.name = 'unlabeled'

    test = pd.read_csv('data/test2.csv')
    test.name = 'test2'

    stopwords_list = stop_words()
    lemmatizer = WordNetLemmatizer()
    # data = [auto_labeled, hand_labeled, unlabeled]
    data = [test]

    preprocessing(data, stopwords_list, lemmatizer)





