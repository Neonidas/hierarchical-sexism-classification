import pandas as pd
from pathlib import Path

data = pd.DataFrame(pd.read_csv("data/edos_labelled_aggregated.csv", sep=','))
train = data[data['split'] == 'train']
dev = data[data['split'] == 'dev']
test = data[data['split'] == 'test']
train = train.drop(columns=['split'])
dev = dev.drop(columns=['split'])
test = test.drop(columns=['split'])

df_train = pd.DataFrame(train)
df_dev = pd.DataFrame(dev)
df_test = pd.DataFrame(test)


filepath_train = Path('data/train.csv')
filepath_dev = Path('data/dev.csv')
filepath_test = Path('data/test.csv')

# df_train.to_csv(filepath_train)
df_dev.to_csv(filepath_dev)
df_test.to_csv(filepath_test)
