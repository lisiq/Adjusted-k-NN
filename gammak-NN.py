import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# reading the dataset Glass (the same dataset as in table Table 1, row no. 6)
df = pd.read_csv('glass0.dat', skiprows=14, names=['col{}'.format(i) for i in range(9)]+['class'])

# removing the white spaces in front of the classes
df['class'] = df['class'].apply(lambda x: x.strip())

# checking which class is scarce
print(df['class'].value_counts())

# the distance function
def euclidean(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
    return s ** 0.5
    
# nearest neighbors with their distances
def nn(k, x, data):
    d = []
    for instance in data:
        d.append(euclidean(x, instance))
    nn = [x for _, x in sorted(zip(d, data))][:k]
    d = sorted(d)[:k]
    return nn, d
    
# merges the neighbours and sorts them by distances
def sortedMerge(nn_p, d_p, nn_m, d_m):
    nn = nn_p + nn_m
    d = d_p + d_m
    nn = [x for _, x in sorted(zip(d, nn))]
    return nn
    
# returns the first k elements of a list
def firstK(k, sM):
    return sM[:k]
    
# classification of a new example with gamma k-NN
def gammakNN(x, positive_data, negative_data, k, gamma, distance_func):
    nn_minus, d_minus = nn(k, x, negative_data) # nearest negative neighbors with their distances
    nn_plus, d_plus = nn(k, x, positive_data) # nearest negative neighbors with their distances
    d_plus = [gamma*i for i in d_plus]
    nn_gamma = firstK(k, sortedMerge(nn_plus, d_plus, nn_minus, d_minus))
    if len([x for x in nn_gamma if x in nn_plus]) >= k/2: # majority vote based on NN_{\gamma}
        return 'positive'
    else:
        return 'negative'
        
# based on the experiments in the paper the data was randomely split with 80% of the data
# for the training data and 20% for the testing data
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size = 0.2, random_state=1)

# the datasets were normalized using min-max normalization where the features are in [-1,1]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
X_train.iloc[:,:-1] = scaler.fit_transform(X_train.iloc[:,:-1])
X_test.iloc[:,:-1] = scaler.transform(X_test.iloc[:,:-1])

# Hyperparameters were tuned with a 10-fold closs-validation over the training set
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# splitting the training set into 10 folds
kf = KFold(n_splits=10)

# we save the F-measure from all the iterations 
f_measure = []

# gamma was selected from the interval [0, 1] using a step of 0.1
for gamma in list(np.arange(0,1,.1)):
    Y = []
    for train_index, test_index in kf.split(X_train):
        X = X_train.iloc[test_index,:] # test set within the cross-validation
        positive_data = X_train.iloc[train_index,:][X_train.iloc[train_index,:]['class'] == 'positive'].iloc[:,:-1].values.tolist()
        negative_data = X_train.iloc[train_index,:][X_train.iloc[train_index,:]['class'] == 'negative'].iloc[:,:-1].values.tolist()
        y = [] # predicted class for the test set
        for x in X.iloc[:,:-1].values.tolist():
            y.append(gammakNN(x, positive_data, negative_data, 3, gamma, euclidean))
        Y.append(f1_score(y, X.iloc[:,-1].values.tolist(),pos_label="positive"))
    f_measure.append(np.mean(Y)) 

# selecting the best gamma that offered the best result
best_gamma = [x for _, x in sorted(zip(f_measure, list(np.arange(0,1,.1))))][-1]   

# chosen gamma from hyperparameter tuning 
print("Best gamma for this dataset: {}".format(best_gamma))

# now training with the whole training set using the best gamma
# in the experiments k was 3
positive_data = X_train[X_train['class'] == 'positive'].iloc[:,:-1].values.tolist()
negative_data = X_train[X_train['class'] == 'negative'].iloc[:,:-1].values.tolist()

for experiment in range(5):
    y = []
    for x in X_test.iloc[:,:-1].values.tolist():
        y.append(gammakNN(x, positive_data, negative_data, 3, best_gamma, euclidean))
        
print('F-measure is: {}'.format(f1_score(y, X_test.iloc[:,-1].values.tolist(),pos_label="positive")))
