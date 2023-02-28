import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

with st.container() :
    st.title('Logistic Regression')
    st.write("Logistic regression, also known as logit regression or logit model, is a mathematical model used in statistics to estimate (guess) the probability of an event occurring having been given some previous data. Logistic regression works with binary data, where either the event happens (1) or the event does not happen (0). So given some feature x it tries to find out whether some event y happens or not. So y can either be 0 or 1. In the case where the event happens, y is given the value 1. If the event does not happen, then y is given the value of 0. For example, if y represents whether a sports team wins a match, then y will be 1 if they win the match or y will be 0 if they do not. This is known as binomial logistic regression. There is also another form of logistic regression which uses multiple values for the variable y. This form of logistic regression is known as multinomial logistic regression.")
    st.write("Logistic regression uses the logistic function to find a model that fits with the data points. The function gives an 'S' shaped curve to model the data. The curve is restricted between 0 and 1, so it is easy to apply when y is binary. Logistic regression can then model events better than linear regression, as it shows the probability for y being 1 for a given x value. Logistic regression is used in statistics and machine learning to predict values of an input from previous test data. ")
iris = datasets.load_iris()
X = iris.data
Y = iris.target
x = st.sidebar.slider('Test Size', min_value = 0.01, max_value = 0.99)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = x)
clf = LogisticRegression()
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
