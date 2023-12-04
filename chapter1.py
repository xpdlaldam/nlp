##### 07/09/2023
#### Packages
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


#### 1.2.2 TF expressions
one_hot_vectorizer = CountVectorizer(binary=True)

corpus = [
        'apple pear', 
        'banana apple',
          ]
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
one_hot

vocab = one_hot_vectorizer.get_feature_names()
vocab

sns.heatmap(one_hot, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence 1', 'Sentence 2'])

# plt.savefig('1-04.png', dpi=300)
plt.show()


#### 
import torch
torch.Tensor(2, 3)

x = torch.arange(6).view(2, 3)
row_indices = torch.arange(2).long()
row_indices
col_indices = torch.LongTensor([0,1])
x[row_indices, col_indices]


#### 3.3.3
import torch
import torch.nn as nn

bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()

probs = sigmoid(torch.randn(4, 1, requires_grad = True))

targets = torch.tensor([1, 0, 1, 0], dtype = torch.float32).view(4, 1)

loss = bce_loss(probs, targets)
