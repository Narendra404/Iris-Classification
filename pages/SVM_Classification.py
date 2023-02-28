import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

with st.container() :
    st.title('SVM Classification')
    st.write("In machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. Developed at AT&T Bell Laboratories by Vladimir Vapnik with colleagues (Boser et al., 1992, Guyon et al., 1993, Cortes and Vapnik, 1995,[1] Vapnik et al., 1997[citation needed]) SVMs are one of the most robust prediction methods, being based on statistical learning frameworks or VC theory proposed by Vapnik (1982, 1995) and Chervonenkis (1974). Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). SVM maps training examples to points in space so as to maximise the width of the gap between the two categories. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.")
iris = datasets.load_iris()
X = iris.data
Y = iris.target
x = st.sidebar.slider('Test Size', min_value = 0.01, max_value = 0.99)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = x)
clf = SVC()
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
st.sidebar.metric(label = ':red[Accuracy]', value = score)
Y_preds = clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_preds)

fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap="Blues", ax = ax)
with st.container() :
    st.markdown('#')
    st.header('Confusion Matrix')
    st.write(fig)
