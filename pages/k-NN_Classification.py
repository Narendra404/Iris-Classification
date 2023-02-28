import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

with st.container() :
    st.title('k-Nearest Neighbors Classification')
    st.write("In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric supervised learning method first developed by Evelyn Fix and Joseph Hodges in 1951,[1] and later expanded by Thomas Cover.[2] It is used for classification and regression. In both cases, the input consists of the k closest training examples in a data set. The output depends on whether k-NN is used for classification or regression.")
    st.write("In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.")
iris = datasets.load_iris()
X = iris.data
Y = iris.target
x = st.sidebar.slider('Test Size', min_value = 0.01, max_value = 0.99)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = x)
clf = KNeighborsClassifier()
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
st.sidebar.metric(label = ':red[Accuracy]', value = score)
Y_preds = clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_preds)

fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(cm, annot=True, annot_kws={"size": 25}, fmt='d', cmap="Blues", ax = ax)
with st.container() :
    st.markdown('#')
    st.header('Confusion Matrix')
    st.write(fig)
