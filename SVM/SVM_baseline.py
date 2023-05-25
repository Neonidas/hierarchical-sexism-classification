import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sys
from datetime import datetime
import pickle
import bz2

analyzer = str(sys.argv[1])
n_gram_start = int(sys.argv[2][0])
n_gram_end = int(sys.argv[2][2])
n_gram_range = (n_gram_start, n_gram_end)
task = str(sys.argv[3])
now = datetime.now()
train = pd.DataFrame(pd.read_csv("../data/train.csv"))
dev = pd.DataFrame(pd.read_csv("../data/dev.csv"))
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
    dev_label = dev['label_sexist']
if task == "b":
    print("Task b")
    train_label = train['label_category']
    test_label = test['label_category']
    dev_label = dev['label_category']
if task == "c":
    print("Task c")
    train_label = train['label_vector']
    test_label = test['label_vector']
    dev_label = dev['label_vector']

print("Training model....")
clf = SVC()
clf.fit(train_data,train_label)
print("Training complete!")

filename = "models/SVM_model_" +analyzer + "_" + str(n_gram_start) + "_" + str(n_gram_range) + "_" + task + "_" + now.strftime("%d_%m_%y_%H:%M:%S") + ".pt"

ofile = bz2.BZ2File(filename, 'wb')
pickle.dump(clf, ofile)
ofile.close()
print("predicting model...")
predictions = clf.predict(test_data)
print("predicting done")
print(classification_report(test_label,predictions))
