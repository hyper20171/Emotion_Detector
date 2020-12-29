#Given a dataset in tsv format with a resonable size and a sentence, program will be able to detect emotion of given sentence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

#Cleaning the text
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #Subsitutes any non letter word with a space
  review = review.lower() #Makes string all lower case
  review = review.split() #Splits the words up into a list with each index contaning a particular word 
  ps = PorterStemmer() #Object that will be used to make a word ( if able to ) into its most basic form as the other forms don't matter
  all_stopwords = stopwords.words('english') #Gets the words that are irrelavent to helping identify emotion of a sentence 
  #all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #Loops through list of words changing words into its most basic form and removing useless words
  review = ' '.join(review) #Converts list back into a string with spaces between each word
  corpus.append(review) #Adds string to corpus list
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Creating bag of word model and fitting it to our cleaned dataset 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()  #Creates bag of word model
X = cv.fit_transform(corpus).toarray() #Bag of word model adjusted to our list containing words 
y = dataset.iloc[:, -1].values #This gets what emotion of each sentence is
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Creating machine learning model and fitting it on bag of word model
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle = False)

from sklearn.neighbors import KNeighborsClassifier    #Create machine learning model that will be used to identify emotion of given sentence                                
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Using machine learning model to predict emotion of given text 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
