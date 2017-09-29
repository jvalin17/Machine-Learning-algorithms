# E-Mail Spam Filtering
The data is taken from UCI Machine Learning Data Repository.The collection of spam e-mails came from their (Hewlett-Packard Labs)
postmaster and individuals who had filed spam. The collection of non-spam e-mails came from filed work and personal e-mails, 
and hence the word 'george' and the area code '650' are indicators of non-spam.  These are useful when constructing a personalized
spam filter.  One would either have to blind such non-spam indicators or get a very wide collection of non-spam to generate a 
general purpose spam filter.

## Data Attributes and Information
Number of Instances: 4601 (1813 Spam = 39.4%)
Number of Attributes: 58 (57 continuous, 1 nominal class label)
The last column of 'spambase.txt' denotes whether the e-mail is considered as spam (1) or not (0), i.e. unsolicited commercial 
e-mail. Most of the attributes indicate whether a particular word or character was frequently occurring in the e-mail.  The 
run-length attributes (55-57) measure the length of sequences of consecutive capital letters.  For the statistical measures of each
attribute, see the end of this file.  Here are the definitions of the attributes:
48 continuous real [0,100] attributes of type word_freq_WORD = percentage of words in the e-mail that match WORD,
i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail.  
A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

6 continuous real [0,100] attributes of type char_freq_CHAR = percentage of characters in the e-mail that match CHAR,
i.e. 100 * (number of CHAR occurrences) / total characters in e-mail

1 continuous real [1,...] attribute of type capital_run_length_average = average length of uninterrupted sequences of capital
letters

1 continuous integer [1,...] attribute of type capital_run_length_longest = length of longest uninterrupted sequence of capital
letters

1 continuous integer [1,...] attribute of type capital_run_length_total = sum of length of uninterrupted sequences of capital
letters = total number of capital letters in the e-mail

1 nominal {0,1} class attribute of type spam = denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited
commercial e-mail.  

Missing Attribute Values: None
Class Distribution:
Spam 1813 (39.4%)
Non-Spam 2788 (60.6%)

## Method

The attributes are continuous so I have considered Gaussian Naïve Bayes for each of the attribute in the data i.e.
P (A_i|C_j) = 1/√(2πσ^2 )*e^(〖(A_i-μ_(y))〗^2/(2*σ_ij^2 )) where σ  is variance and μ is mean A_i is attribute and C_j is Class 

So I am taking mean and variance of each of the attribute with a particular class (1 and 0) and then using it in each column
(attribute) of the test data. This gives the probability whether the test data will be spam or not. The one that has higher 
probability will be classified. 
I have applied concept of split ratio as there is no test data. The split percentage for training data has to be provided by the user.

## Flow
Run python program.
It will ask for input from user for split percentage and give output.  
It will make two files that is train.txt and test.txt.  
It will run on train data and test data and predicts the output.  

## Analysis
The splitting of data is randomly done using functionrandom.sample() so splitting is unbiased and hence it gives different
accuracies every time you run the code.  

Prior Probabilities also change due different data points.

The accuracies are similar but not same. I have ran the code for same training data split percentage for
3 times each and achieved different accuracies.  

Maximum Accuracy on an average is 82.6% which is achieved when split ratio is 75:25 (train:test).

