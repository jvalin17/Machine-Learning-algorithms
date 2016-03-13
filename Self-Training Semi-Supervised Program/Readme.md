###Problem Statement: Predict gender for given data which conisits of age sex and weight as attributes.

####Data
Ds - set of records given in data-set
Du - set of unlabeled records given in data-set.
Dt - set of test records given in data-set.

####Files:  
ml3_Ds.csv - Ds
ML_p3.csv -Ds
MLp.csv -Ds
test_ml3.csv - Du
ml3_2.csv - Dt

####Structure:  
ml3_Ds, ML_p3, MLp have same content but two of the files get updated in one of the code and all files are used so there are 3 differently named same files. 

Part a:
Input : 'ML_p3.csv'
Program Name: MLP3.py
Language: Python 

Self Training Semi Supervised Learning Algorithm  

It takes input from ML_p3.csv and runs self training logistic regression on it. So it updates the input file when it adds unlabeled data in it.

So if you want to implement the code again (more than once) ,clear the data from input file that is ML_p3.csv and you need to copy all rows from ml3_Ds.csv and paste the rows in input file.   


Part b:  
Program Name: mlP3.2.py  
Naive Bayes Classifier  

Input : 'ml3_Ds.csv'

Language: Python

It just takes input from the input file and tests the results on ml3_2.csv which is Dt in the given data-set. The accuracy is calculted in the program itself.

Semi-supervised Learning: Self Training Logistic Algorithm (Same code as above) 
Input: 'MLp.csv'  

It takes input from input file and applies the algorithm same as described above and it has same limitation for running it again.  

At the end accuracy from both classifiers are printed on screen.
