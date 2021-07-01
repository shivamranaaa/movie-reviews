from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import pickle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle
import re

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
word_dict=pickle.load(open("bow.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
			review = request.form['review']
			rev=review
	def clean(text):
		cleaned = re.compile(r'<.*?>')
		return re.sub(cleaned,'',text)

	def is_special(text):
		rem = ''
		for i in text:
			if i.isalnum():
				rem = rem + i
			else:
				rem = rem + ' '
			return rem

	def to_lower(text):
		return text.lower()

	def rem_stopwords(text):
		stop_words = set(stopwords.words('english'))
		words = word_tokenize(text)
		return [w for w in words if w not in stop_words]

	def stem_txt(text):
		ss = SnowballStemmer('english')
		return " ".join([ss.stem(w) for w in text])

	f1 = clean(rev)
	f2 = is_special(f1)
	f3 = to_lower(f2)
	f4 = rem_stopwords(f3)
	f5 = stem_txt(f4)

	bow,words = [],word_tokenize(f5)
	for word in words:
		bow.append(words.count(word))

	inp = []
	for i in word_dict:
		inp.append(f5.count(i[0]))

	pred = clf.predict(np.array(inp).reshape(1,1000))

	if pred==1:
		output="Positive"
	else:
		output="Negative"

	return render_template('index.html',prediction = output)

if __name__ == '__main__':
	app.run(debug=True)