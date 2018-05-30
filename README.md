#Spam Classifier Made using SVM model

dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset/data

Pre Processing done:
Stemming - Porter Stemer
Removed all the stop words
Used regulatr exression to replace all the email address in sms to string 'email', all the web address to string 'httpadr' and all the number to string 'number'
Removed all the sms string length equal to one.

Test data/Train data ratio = 0.33

Model used:
SVM - Support Vector machine
Kernel - Gaussian Kernel

Best Model:
C = 600
Train Accuracy      0.997321
Test Accuracy       0.985318
Test Recall         0.900794
Test Precision      0.991266

Confusion Matrix:
	        Predicted 0	Predicted 1
Actual 0	1584	    3
Actual 1	24	        228
