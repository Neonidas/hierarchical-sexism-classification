import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import sys
import pickle
import bz2

analyzer = str(sys.argv[1])
n_gram_start = int(sys.argv[2][0])
n_gram_end = int(sys.argv[2][2])
n_gram_range = (n_gram_start, n_gram_end)
task = str(sys.argv[3])
filename = sys.argv[4]
train = pd.DataFrame(pd.read_csv("../data/train.csv"))
test = pd.DataFrame(pd.read_csv("../data/test.csv"))

vectorizer = TfidfVectorizer(max_features=50000, analyzer=analyzer, ngram_range=n_gram_range)
vec_train_data =  vectorizer.fit_transform(train['text'])
vec_train_data = vec_train_data.toarray()
vec_test_data = vectorizer.transform(test['text']).toarray()

train_data = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
test_data = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())

if task == "a":
    print("Task a")
    train_label = train['label_sexist']
    test_label = test['label_sexist']
if task == "b":
    print("Task b")
    train_label = train['label_category']
    test_label = test['label_category']
if task == "c":
    print("Task c")
    train_label = train['label_vector']
    test_label = test['label_vector']
ifile = bz2.BZ2File(filename,'rb')

clf = pickle.load(ifile)
ifile.close()
print("predicting model...")
predictions = clf.predict(test_data)
print("predicting done")
print(classification_report(test_label,predictions))
