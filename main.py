from __future__ import division
import train
import test
import matplotlib.pyplot as plt

def report(result_list):
	pr0 = precision(result_list, 0)
	pr1 = precision(result_list, 1)
	return max(pr0, pr1)

def precision(result_list, c):
	return (result_list[c][c])/(result_list[c][c]+result_list[1-c][c])

def main():
	total_input_size = 28379
	ratio_list = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
	accuracy_list = []
	input_size_list = []
	for ratio in ratio_list:
		train.train_classifier(ratio)
		test.test_classifier()
		print ' === Results for classification with Laplace smooting === \n'
		accuracy_list.append(report(test.result_list))
		input_size_list.append(total_input_size * ratio)

	plt.plot(input_size_list, accuracy_list, 'ro')
	plt.show()

main()
