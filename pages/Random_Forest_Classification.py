import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

with st.container() :
    st.title('Random Forest Classification')
    st.write("Random forest is a statistical algorithm that is used to cluster points of data in functional groups. When the data set is large and/or there are many variables it becomes difficult to cluster the data because not all variables can be taken into account, therefore the algorithm can also give a certain chance that a data point belongs in a certain group. ")
iris = datasets.load_iris()
X = iris.data
Y = iris.target
x = st.sidebar.slider('Test Size', min_value = 0.01, max_value = 0.99)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = x)
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
st.sidebar.metric(label = ':red[Accuracy]', value = score)
Y_preds = clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_preds)

fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap="Blues", ax = ax)
ax.set_title('Confusion Matrix')
with st.container() :
    st.markdown('#')
    st.header('Confusion Matrix')
    st.write(fig)
