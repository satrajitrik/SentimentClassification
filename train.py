import glob, os
import re
from collections import Counter

class_dictionary = [{}, {}]
number_of_documents = [0, 0]
number_of_terms = [0, 0]
conditional_probability = {}

def words(text):
	return re.findall(r'\w+', text)

def save_sparse_matrix(word_dictionary, file):
	fileobj = open('/Users/satrajitmaitra/Documents/fall-18/Machine Learning/assignments/movie review data/sparse_matrix.txt','a')
	for word in word_dictionary:
		fileobj.write(word + ", " + file + ", " + str(word_dictionary[word]) + "\n")
	fileobj.close()

def read_files_from_directory(directory, label, ratio):
	os.chdir(directory)
	length = len(glob.glob("*.txt")) * ratio
	for file in glob.glob("*.txt"):
		word_dictionary = Counter(words(open(file).read()))
		save_sparse_matrix(word_dictionary, file)
		build_class_dictionary(word_dictionary, label)
		number_of_documents[label] += 1
		length -= 1
		if length == 0:
			break

def build_class_dictionary(word_dictionary, label):
	for word in word_dictionary:
		number_of_terms[label] += 1
		if word in class_dictionary[label]:
			class_dictionary[label][word] += 1
		else:
			class_dictionary[label][word] = 1

def calculate_conditional_probability():
	global conditional_probability
	conditional_probability = {}
	for index, words in enumerate(class_dictionary):
		for word, word_count in words.items():
			if word not in conditional_probability:
				conditional_probability[word] = {0: 0, 1: 0}
				conditional_probability[word][1-index] = 1 / float(number_of_terms[index] + len(class_dictionary[index]))
			conditional_probability[word][index] = (word_count + 1) / float(number_of_terms[index] + len(class_dictionary[index]))

def train_classifier(ratio):
	current_working_directory = os.getcwd()
	read_files_from_directory("/Users/satrajitmaitra/Documents/fall-18/Machine Learning/assignments/movie review data/pos/", 1, ratio)
	os.chdir(current_working_directory)
	read_files_from_directory("/Users/satrajitmaitra/Documents/fall-18/Machine Learning/assignments/movie review data/neg/", 0, ratio)
	os.chdir(current_working_directory)
	calculate_conditional_probability()
