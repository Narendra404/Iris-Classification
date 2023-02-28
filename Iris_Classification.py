import streamlit as st 
import json
import requests
from streamlit_lottie import st_lottie

with st.container() :
    st.title('Iris Classification')
    st.header('A simple Machine Learning Project with the use of the famous Iris flower dataset.')
    st.write('This project involves the result of four different ML classification algorithms. The test size for each algorithm can be changed and the accuracy and the confusion matrix is displayed.')

with st.container() :
    st.markdown('##')
    st.header('About Dataset')
    st.markdown('#')
    st.subheader('Context')
    st.write("The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.")
    st.write('This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines')
    st.markdown('#')
    st.subheader('Content')
    st.write('The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).')
    st.markdown('#')
    st.subheader('Acknowledgements')
    st.write('This dataset is free and is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)')

with st.sidebar.container() :
    url = requests.get("https://assets1.lottiefiles.com/packages/lf20_zeoujfto.json")
    url_json = dict()
    if url.status_code == 200 :
        url_json = url.json()
    else:
        print("Error in the url")
    st_lottie(url_json)

