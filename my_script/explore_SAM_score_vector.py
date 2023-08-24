
from glob import glob
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import discriminant_analysis, manifold, linear_model
from sklearn.model_selection import train_test_split

def extract_score_vector(path, histogram=False):
	ls = sorted(os.listdir(path))
	pred_ious = []
	stability_scores = []
	for _, f in enumerate(ls):
		if _ in [294, 307, 347]: # 3 cases with nan in the data, skip
			continue
		with open(path+'/'+f+'/'+'metadata.csv', 'r') as csvfile:
			reader = csv.reader(csvfile)
			pred_iou = []
			stability_score = []
			for i, row in enumerate(reader):
				if i == 0:
					continue
				# pred_iou in the 8th column, stability score in the 9th column

				pred_iou.append(float(row[8]))
				stability_score.append(float(row[9]))
			pred_ious.append(pred_iou)
			stability_scores.append(stability_score)

	print(len(pred_ious), len(stability_scores))
	
	if histogram:
		# # calculate bins
		# _, pred_iou_bins = np.histogram(np.concatenate(pred_ious))
		# _, stability_score_bins = np.histogram(np.concatenate(stability_scores))
		# print(pred_iou_bins, stability_score_bins)

		# make bins universal for all data according to the statistics
		pred_iou_bins = np.linspace(0.88, 1.11, 11)
		stability_score_bins = np.linspace(0.949, 1.0, 11)

		# apply bins to pred_iou and stability_score
		pred_iou_features = []
		for pred_iou in pred_ious:
			pred_iou_features.append(np.histogram(pred_iou, pred_iou_bins)[0])
		stability_score_features = []
		for stability_score in stability_scores:
			stability_score_features.append(np.histogram(stability_score, stability_score_bins)[0])
		return np.concatenate((pred_iou_features, stability_score_features), axis=1)
	else:
		return np.concatenate((np.array(pred_ious), np.array(stability_scores)), axis=1)

def dim_reduce_for_2sets(dim_reduce_method, histogram, run_name='', first_set_path='SA-430-resized_rle', second_set_path='failure_cases_rle'):
	first_features = extract_score_vector(first_set_path, histogram=histogram)
	second_features = extract_score_vector(second_set_path, histogram=histogram)
	print(first_features.shape, second_features.shape)
	features = np.concatenate((first_features, second_features), axis=0)
	print(features.shape)
	
	if dim_reduce_method == 'pca':
		method = PCA(n_components=2)
		method.fit(features)
		first_features = method.transform(first_features)
		second_features = method.transform(second_features)
		plt.scatter(first_features[:, 0], first_features[:, 1], c='r', marker='o', label='SA-430')
		plt.scatter(second_features[:, 0], second_features[:, 1], c='b', marker='x', label='failure cases')
	elif dim_reduce_method == 'lda':
		method = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
		Y = np.concatenate((np.ones(first_features.shape[0]), np.zeros(second_features.shape[0])), axis=0)
		method.fit(features, Y)
		first_features = method.transform(first_features)
		second_features = method.transform(second_features)
		plt.scatter(first_features[:, 0], np.zeros_like(first_features[:, 0]), c='r', marker='o', label='SA-430')
		plt.scatter(second_features[:, 0], np.zeros_like(second_features[:, 0]), c='b', marker='x', label='failure cases')
	elif dim_reduce_method == 'tsne':
		method = manifold.TSNE(n_components=2, random_state=0, perplexity=30)
		features = method.fit_transform(features)
		plt.scatter(features[:first_features.shape[0], 0], features[:first_features.shape[0], 1], c='r', marker='o', label='SA-430')
		plt.scatter(features[first_features.shape[0]:, 0], features[first_features.shape[0]:, 1], c='b', marker='x', label='failure cases')
	print(first_features.shape, second_features.shape)
	plt.legend()
	plt.savefig(run_name+'_'+dim_reduce_method+'.png')

def logreg_for_2sets(use_histogram, first_set_path='SA-430-resized_rle', second_set_path='failure_cases_rle'):
	first_features = extract_score_vector(first_set_path, use_histogram)
	first_labels = np.zeros(first_features.shape[0])
	second_features = extract_score_vector(second_set_path, use_histogram)
	second_lables = np.ones(second_features.shape[0])
	print(first_features.shape, second_features.shape)
	# split the data into training and testing
	first_features_train, first_features_test, first_labels_train, first_labels_test = train_test_split(first_features, first_labels, test_size=0.2, random_state=0)
	second_features_train, second_features_test, second_labels_train, second_labels_test = train_test_split(second_features, second_lables, test_size=0.2, random_state=0)
	print('trainset size: ', first_features_train.shape[0], second_features_train.shape[0])
	print('testset size: ', first_features_test.shape[0], second_features_test.shape[0])

	train_x = np.concatenate((first_features_train, second_features_train), axis=0)
	train_y = np.concatenate((first_labels_train, second_labels_train), axis=0)

	from imblearn.over_sampling import RandomOverSampler
	ros = RandomOverSampler(random_state=0)
	X_resampled, y_resampled = ros.fit_resample(train_x, train_y)

	logreg = linear_model.LogisticRegressionCV(Cs=np.logspace(-4,4,20), cv=5, n_jobs=-1)
	logreg.fit(X_resampled, y_resampled)
	print('text acc on first (easy) dataset: ', logreg.score(first_features_test, first_labels_test))
	print('text acc on second (hard) dataset: ', logreg.score(second_features_test, second_labels_test))

	# inspect the model
	print(logreg.coef_)
	print(first_features[:10])
	print(second_features[:10])



if __name__ == '__main__':
	# dim_reduce_for_2sets('lda', False, 'nothresh', 'SA-430-resized_rle_nothresh', 'failure_cases_nothresh')
	# logreg_for_2sets(True, 'SA-430-resized_rle', 'failure_cases_rle')



