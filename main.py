from distutils import text_file
from pyexpat import model
import numpy as np
from pyparsing import Regex
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import nltk
import time
import flask
from nltk.corpus import stopwords
import string
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix,f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request
from werkzeug.utils import secure_filename
import io
from distutils.core import setup

#Initialize flask app
app = Flask(__name__)

#Request methods and python script
@app.route('/',methods=['POST','GET'])
def script_run():
    if request.method == 'POST':
        data = request.form['text']
        if data != '':
            nltk.download('stopwords')
            def load_tweets_data(name):
                csv_path = os.path.join("", name)
                return pd.read_csv(csv_path)
            
            
        
            text2 = open("sample.txt", "r+",encoding='utf-8')
            text2.write(data)
            text2.close()
            text2 = open("sample.txt", "r+",encoding='utf-8')
            
            print(data)
            text1 = pd.read_csv("sample.txt",names=['news'])
            text2.truncate()
            text2.close()
            print(text1)
            

            #Replace every blank with non blank
            #text1 = data.replace(" ", "")


            #Punctuation removal for string
            #text1 = data.translate(str.maketrans('', '', string.punctuation)) #Use maketrans from string librady to transform the given to text withour punctuation
            
            #Transform string to lower case 
            #text1 = data.lower()

            #Show text format on terminal
            trainTweets = load_tweets_data("train.csv")
            testTweets = load_tweets_data("test.csv")

            print(trainTweets.shape)

            print(testTweets.shape)

            trainTweets.info()

            print(trainTweets.head())
            
            text1=text1.fillna('')
            trainTweets=trainTweets.fillna('')
            testTweets=testTweets.fillna('')
           
            
            print(trainTweets)
            trainTweets['total'] = trainTweets['title']+' '+trainTweets['author']
            print(trainTweets)
            testTweets['total']=testTweets['title']+' '+testTweets['author']

            print(trainTweets.head())
            X_train = trainTweets.drop('label',axis=1)
            y_train=trainTweets['label']
            print(X_train)

            #X_test = testTweets.drop('label',axis=1)
            #y_test=testTweets['label']

            print(X_train.shape)
            print(y_train.shape)
            print(X_train.head())
            print(y_train.head())



            # Convert to lowercase
            #text1=text1.apply(lambda x: x.lower())
            trainTweets['total'] = trainTweets['total'].apply(lambda x: x.lower())
            testTweets['total'] = testTweets['total'].apply(lambda x: x.lower())
            
            trainTweets.head()



            def punctuation_removal(text):
                all_list = [char for char in text if char not in string.punctuation]
                clean_str = ''.join(all_list)
                return clean_str

            text1 = text1.apply(punctuation_removal)
            trainTweets['total'] = trainTweets['total'].apply(punctuation_removal)
            testTweets['total'] = testTweets['total'].apply(punctuation_removal)
           
            
            
            print(trainTweets.head())


            stop = stopwords.words('english')

            text1 = text1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            trainTweets['total'] = trainTweets['total'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            testTweets['total'] = testTweets['total'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            

            print(trainTweets.groupby(['label'])['total'].count())
            #print(testTweets.groupby(['label'])['total'].count())

            trainTweets.groupby(['label'])['total'].count().plot(kind='bar')
            locs, labels = plt.xticks()
            plt.xticks(np.arange(0, 1, step=0.2))
            plt.xticks(np.arange(2), ('True', 'False'))
            plt.xticks(rotation=0)
            plt.title("True and False News")
            plt.xlabel("")
            plt.show()

            

            Tfidf_vect = TfidfVectorizer(max_features=3600, min_df=2, max_df=0.5, ngram_range=(1,1))

            Tfidf_vect.fit(trainTweets['total'])
            train_x_tfidf = Tfidf_vect.transform(trainTweets['total'])
            Tfidf_vect.fit(testTweets['total'])
            test_x_tfidf = Tfidf_vect.transform(testTweets['total'])
            text_tfidf = Tfidf_vect.transform(text1)
            #new text
            print(text1)
            print(trainTweets)
            

            

            labels = ['True Neg','False Pos','False Neg','True Pos']
            categories = ['Fake', 'Real']


            print(testTweets)
            

            X_train1,X_test1,Y_train1,Y_test1 = train_test_split(train_x_tfidf,y_train,test_size = 0.3, stratify=y_train, random_state=2)
            X_train1000,X_test1000,Y_train1000,Y_test1000 = train_test_split(train_x_tfidf,y_train,test_size = 1000, stratify=y_train, random_state=2)

            
            def func_LogisticRegression(X_train1,X_test1,Y_train1,Y_test1,size,text):

                #Logistic Regression
                model = LogisticRegression()
                model.fit(X_train1, Y_train1)
                
                
                
                #dokimi sto train set
                X_train_prediction = model.predict(X_train1)
                training_data_accuracy = accuracy_score(X_train_prediction, Y_train1)

                print(training_data_accuracy)

                reg_conf_mat= confusion_matrix(y_true=Y_train1,y_pred=X_train_prediction)
                reg_classification_report=classification_report(Y_train1,X_train_prediction)

                print("Logistic Regression "+size+" Train-set")
                print(reg_conf_mat)
                print(reg_classification_report)


                plot_confusion_matrix(model,X_train1,Y_train1)
                plt.title(size+" Train Set   Logistic Regression")
                plt.show( )


                #dokimi sto test set
                X_test_prediction = model.predict(X_test1)
                test_data_accuracy = accuracy_score(X_test_prediction, Y_test1)
                print(test_data_accuracy)

                reg_conf_mat= confusion_matrix(y_true=Y_test1,y_pred=X_test_prediction)
                reg_classification_report=classification_report(Y_test1,X_test_prediction)

                print("Logistic Regression "+size+" Test-set")
                print(reg_conf_mat)
                print(reg_classification_report)

                plot_confusion_matrix(model,X_test1,Y_test1)
                plt.title(size+" Test set Logistic Regression")
                plt.show( )

                text_pred = model.predict(text)
                print("lOGISTIC REGRESSION TEXT PREDICT:",text_pred)
                
                
                #dokimi sto input set


            def func_SVM(X_train1,X_test1,Y_train1,Y_test1,size,text):
                #SVM
                modelSvm = SVC()
                modelSvm.fit(X_train1, Y_train1)

                #dokimi sto train set
                X_train_prediction = modelSvm.predict(X_train1)
                training_data_accuracySvm = accuracy_score(X_train_prediction, Y_train1)

                print(training_data_accuracySvm)

                Svm_conf_mat= confusion_matrix(y_true=Y_train1,y_pred=X_train_prediction)
                Svm_classification_report=classification_report(Y_train1,X_train_prediction)

                print("Support Vector Machine "+size+" Train-set")
                print(Svm_conf_mat)
                print(Svm_classification_report)


                plot_confusion_matrix(modelSvm,X_train1,Y_train1)
                plt.title(size+" Train Set SVM")
                plt.show( )

                #dokimi sto test set
                X_test_prediction = modelSvm.predict(X_test1)
                test_data_accuracySvm = accuracy_score(X_test_prediction, Y_test1)
                print(test_data_accuracySvm)

                Svm_conf_mat= confusion_matrix(y_true=Y_test1,y_pred=X_test_prediction)
                Svm_classification_report=classification_report(Y_test1,X_test_prediction)

                print("Support Vector Machine "+size+" Test-set")
                print(Svm_conf_mat)
                print(Svm_classification_report)

                plot_confusion_matrix(modelSvm,X_test1,Y_test1)
                plt.title(size+" Test set SVM")
                plt.show( )

                
                text_pred=modelSvm.predict(text)
                print("SVM TEXT PREDICT:",text_pred)



            

            func_LogisticRegression(X_train1,X_test1,Y_train1,Y_test1,"Random",text_tfidf)
            tic=time.perf_counter() #arxh training me to logistic
            func_SVM(X_train1,X_test1,Y_train1,Y_test1,"Random",text_tfidf)
            toc=time.perf_counter()
            timer=toc-tic
            
            

            func_LogisticRegression(X_train1000,X_test1000,Y_train1000,Y_test1000,"1000",text_tfidf)#Den xrhsiompoioume to apotelesma tou text apo edw
            #Metrisi xronou pou xreiazetai gia thn ekpaideush tou montelou
            tic_1k=time.perf_counter() #arxh training me to logistic
            func_SVM(X_train1000,X_test1000,Y_train1000,Y_test1000,'1000',text_tfidf) #Den xrhsiompoioume to apotelesma tou text apo edw
            toc_1k=time.perf_counter()

            timer_1k=toc_1k-tic_1k
            averagePerNews=timer_1k/1000
            print("Xronos ekpaideushs gia ola ta keimena:",timer,"seconds")
            print("Xronos ekpaideushs gia 1000 keimena:",timer_1k,"seconds")
            print("Xronos gia neo keimeno:",averagePerNews,"seconds")



            print("END")

            return redirect(url_for('script_run'))
        if text1 == '':
            return render_template('index.html')
    else:
       return render_template('index.html')



if __name__ == '__main__':
    app.run(debug = True)