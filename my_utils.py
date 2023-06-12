from glob import glob
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import discriminant_analysis, manifold, linear_model
from sklearn.model_selection import train_test_split

def delete_png_files():
	# delete png files reccursively from cur dir
	for f in glob('./**/*.png', recursive=True):
		print(f)
		os.remove(f)	

if __name__ == '__main__':
	delete_png_files()