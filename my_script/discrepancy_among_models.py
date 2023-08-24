import itertools
import glob
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


from my_utils import load_segmentation_and_info, greedy_iou

models = ['sam', 'oneformer', 'mask2former', 'maskformer'] 
similarity_path = '/home/yjl/data/val2017_similarity/'
segment_area_thresh_low = 0.04
segment_area_thresh_high = 0.63

combinations = list(itertools.combinations(models, 2))

img_name_ls = sorted(glob.glob('/home/yjl/data/val2017_oneformer_results/*.pkl'))
img_name_ls = [img_name.split('/')[-1].split('.')[0] for img_name in img_name_ls]
# print(len(img_name_ls)) # 4990

for img_name in tqdm(img_name_ls):
	try:
		with open(similarity_path+img_name+'.json', 'r') as file:
			data = json.load(file)
	except FileNotFoundError:
		# If the file doesn't exist, initialize an empty data structure
		data = {}
	
	for model1, model2 in combinations:
		
		# load segmentation
		if model1 == 'sam':
			# get the sam segmentation of the image
			segmentation1 = glob.glob(f'/home/yjl/data/val2017_{model1}_results/' + img_name + '/*.png')
			segmentation1 = [np.array(Image.open(sam_segment))/255 for sam_segment in segmentation1]
			segmentation1 = [torch.from_numpy(sam_segment).bool() for sam_segment in segmentation1]
			
		else:
			segmentation1, _ = load_segmentation_and_info(f'/home/yjl/data/val2017_{model1}_results/{img_name}', return_list=True)
		segmentation2, _ = load_segmentation_and_info(f'/home/yjl/data/val2017_{model2}_results/{img_name}', return_list=True)
		print(len(segmentation1), len(segmentation2))

		# filter out small and large segments
		if segment_area_thresh_low != 0 or segment_area_thresh_high != 1:
			area_filter = True
			segmentation1 = [sam_segment for sam_segment in segmentation1 if \
				segment_area_thresh_low < sam_segment.sum().item()/sam_segment.numel() < segment_area_thresh_high]
			segmentation2 = [sam_segment for sam_segment in segmentation2 if \
				segment_area_thresh_low < sam_segment.sum().item()/sam_segment.numel() < segment_area_thresh_high]
		print(len(segmentation1), len(segmentation2))
		
		# get iou and save
		various_sim_for_2models = dict() 
		try: # some segment results are empty 
			if area_filter:
				various_sim_for_2models[f'greedyiou_thresh{segment_area_thresh_low}-{segment_area_thresh_high}'] = greedy_iou(segmentation1, segmentation2, area_weighted=False)
				various_sim_for_2models[f'greedyiou_thresh{segment_area_thresh_low}-{segment_area_thresh_high}_areaWeighted'] = greedy_iou(segmentation1, segmentation2, area_weighted=True)
			else:
				various_sim_for_2models['greedyiou'] = greedy_iou(segmentation1, segmentation2)
				# various_sim_for_2models['greedyiou_top5'] = greedy_iou(segmentation1, segmentation2, top_n=5)
				various_sim_for_2models['greedyiou_top10'] = greedy_iou(segmentation1, segmentation2, top_n=10)
				# various_sim_for_2models['greedyiou_top15'] = greedy_iou(segmentation1, segmentation2, top_n=15)
				# various_sim_for_2models['greedyiou_top20'] = greedy_iou(segmentation1, segmentation2, top_n=20)
				# various_sim_for_2models['greedyiou_top30'] = greedy_iou(segmentation1, segmentation2, top_n=30)
				various_sim_for_2models['greedyiou_area_weighted'] = greedy_iou(segmentation1, segmentation2, area_weighted=True)

		except Exception as error:
			print("Error in", img_name, model1, model2)
			print(error)
			various_sim_for_2models['greedyiou'] = various_sim_for_2models['greedyiou_top5'] = \
			various_sim_for_2models['greedyiou_top10'] = various_sim_for_2models['greedyiou_top15'] = \
			various_sim_for_2models['greedyiou_top20'] = various_sim_for_2models['greedyiou_top30'] = \
			various_sim_for_2models['greedyiou_area_weighted'] = -1

		data[model1+'_'+model2].update(various_sim_for_2models)

	
	# Save the data
	print(data)
	with open(similarity_path+img_name+'.json', 'w') as file:
		json.dump(data, file)
	print(img_name, "saved")	
