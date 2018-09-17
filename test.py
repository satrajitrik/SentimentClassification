import glob, os
import train
from collections import Counter
import re

result_list = [[0, 0],[0, 0]]

def words(text):
	return re.findall(r'\w+', text)

def read_files_from_directory(directory, label):
	os.chdir(directory)
	for file in glob.glob("*.txt"):
		word_dictionary = Counter(words(open(file).read()))
		classification = classify_text(word_dictionary)
		result_list[label][classification] += 1

def classify_text(word_dictionary):
	p0 = float(train.number_of_documents[0]) / float(train.number_of_documents[0] + train.number_of_documents[1])
	p1 = float(train.number_of_documents[1]) / float(train.number_of_documents[0] + train.number_of_documents[1])

	decision = 0
	for word in word_dictionary:
		if word in train.conditional_probability:
			pi0 = train.conditional_probability[word][0]
			pi1 = train.conditional_probability[word][1]
			numerator_0 = pi0 * p0
			numerator_1 = pi1 * p1
			decision = numerator_0 - numerator_1
	return (0 if decision > 0 else 1)


def test_classifier():
	global result_list
	result_list = [[0,0], [0,0]]
	current_working_directory = os.getcwd()
	read_files_from_directory("/Users/satrajitmaitra/Documents/fall-18/Machine Learning/assignments/movie review data/neg/", 0)
	os.chdir(current_working_directory)
	read_files_from_directory("/Users/satrajitmaitra/Documents/fall-18/Machine Learning/assignments/movie review data/pos/", 1)
	os.chdir(current_working_directory)
