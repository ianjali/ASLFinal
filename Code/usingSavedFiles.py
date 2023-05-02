import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with open('data_list.pickle', 'rb') as f:
    # Use pickle.load() to read the list from the file
    train_data = pickle.load(f)
with open('labels_list.pickle', 'rb') as f:
    # Use pickle.dump() to save the list to the file
    train_labels = pickle.load(f)

with open('test_data.pickle', 'rb') as f:
    # Use pickle.load() to read the list from the file
    test_data = pickle.load(f)
with open('test_labels.pickle', 'rb') as f:
    # Use pickle.dump() to save the list to the file
    test_labels = pickle.load(f)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Fit and transform the labels
encoded_labels = le.fit_transform(train_labels)
encoded_labels_test = le.fit_transform(test_labels)

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.3)

# # Feature scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Encode the labels
# encoder = LabelEncoder()
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.transform(y_test)

# Train the SVM model
svm = SVC(kernel='linear', C=1)
svm.fit(train_data, encoded_labels)

# Evaluate the model
y_pred = svm.predict(train_data)
accuracy = accuracy_score(encoded_labels, y_pred)
print("Accuracy(train): {:.2f}%".format(accuracy*100))

y_pred = svm.predict(test_data)
accuracy = accuracy_score(encoded_labels_test, y_pred)
print("Accuracy(test): {:.2f}%".format(accuracy*100))


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def train_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans


def evaluate_kmeans(kmeans_model, X, y):
    y_pred = kmeans_model.predict(X)
    return adjusted_rand_score(y, y_pred),y_pred
# Train a k-means model
kmeans_model = train_kmeans(train_data, n_clusters=29)

print(f"{'='*3} K Means {'='*3}")
# Evaluate the k-means model
ars,pred_train = evaluate_kmeans(kmeans_model, train_data, train_labels)
print("Adjusted Rand Score:", ars)


ars,pred_tes = evaluate_kmeans(kmeans_model, test_data, test_labels)
print("Adjusted Rand Score:", ars)

# Calculate accuracy score
accuracy = accuracy_score(train_labels,pred_train )
print("Accuracy: {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(test_labels,pred_tes )
print("Accuracy: {:.2f}%".format(accuracy * 100))



