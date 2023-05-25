import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sys
from datetime import datetime
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion

task = str(sys.argv[1])
now = datetime.now()
train = pd.DataFrame(pd.read_csv("../data/train.csv"))
dev = pd.DataFrame(pd.read_csv("../data/dev.csv"))
test = pd.DataFrame(pd.read_csv("../data/test.csv"))

pipe = Pipeline([
    ('feats', FeatureUnion([
        ('tfic_word', TfidfVectorizer(max_features=50000, analyzer='word')),
        ('tfic_char' , TfidfVectorizer(max_features=50000, analyzer='char', ngram_range=(3,6))),
    ])),
    
    ('svm' ,SVC())
])

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
pipe.fit(train['text'],train_label)
print("Training complete!")

#filename = "models/SVM_model_" +"concatted_best" + "_" + now.strftime("%d_%m_%y_%H:%M:%S") + ".pt"
#pickle.dump(clf, open(filename,"wb"))
print("predicting model...")
y = pipe.predict(test['text'])
print("predicting done")
print(classification_report(test_label,y))
