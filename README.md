# Depression-detection

## required libraries

pandas,
os,
pickle,
io,
demoji,
nltk,
re,
string,
random,
numpy,
tensorflow,
fasttextasft,
keras,
matplotlib,
sklearn,
demoji

all of these libraries can be installed through pip

## instructions for data_loading_and_preprocessing

this is a stand alone module which requires the user to enter a dataset location with 2 folder ,preferably labeled positive and negative,with positive folder containing positively identified depressive tweets and the negative folder containing normal tweets

a sample dataset is provided in the data folder ,labeled , dataset_tweets_rm.zip
to use the dataset,extract the above mentioned file and enter the location of the file at the prompt

this module creates a pickle file which contains a dictionary of all the preprocessed tweets along with their labels and vectors




## instructions for model_training
this is a stand alone module which requires the user to enter a pickle file containing the preprocessed tweets from the data_loading_and_preprocessing module,the pickle file should contain a dictionary with atleast 5368 elements and 4 columns ,namely,data,class,tweet,vector

a sample file is provided in the data folder ,labeled , users(1).pickle

this module trains and provides the metrics for the their accuracies
