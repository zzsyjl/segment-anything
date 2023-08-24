import itertools
import glob
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


from my_utils import load_segmentation_and_info, greedy_iou, non_max_suppression

attackers = ['oneformer', 'mask2former', 'maskformer']
similarity_path = '/home/yjl/data/val2017_similarity/'
segment_area_thresh_low = 0
segment_area_thresh_high = 1



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
	
		
		# load segmentation

	# get the sam segmentation of the image
	sam_segments = glob.glob(f'/home/yjl/data/val2017_sam_results/' + img_name + '/*.png')
	sam_segments = [np.array(Image.open(sam_segment))/255 for sam_segment in sam_segments]
	sam_segments = [torch.from_numpy(sam_segment).bool() for sam_segment in sam_segments]
	
	attackers_segments = []
	for attacker in attackers:
		attacker_segments, _ = load_segmentation_and_info(f'/home/yjl/data/val2017_{attacker}_results/{img_name}', return_list=True)
		attackers_segments.extend(attacker_segments)


	# filter out small and large segments
	if segment_area_thresh_low != 0 or segment_area_thresh_high != 1:
		sam_segments_filtered = [sam_segment for sam_segment in sam_segments if \
			segment_area_thresh_low < sam_segment.sum().item()/sam_segment.numel() < segment_area_thresh_high]
		attackers_segments_filtered = [attackers_segment for attackers_segment in attackers_segments if \
			segment_area_thresh_low < attackers_segment.sum().item()/attackers_segment.numel() < segment_area_thresh_high]
		print(len(sam_segments), len(attackers_segments))
	
		attackers_segments_filtered_nms = non_max_suppression(attackers_segments_filtered)
	
	attackers_segments_nms = non_max_suppression(attackers_segments)
		
	# get iou and save
	various_sim_for_2models = dict() 
	try: # some segment results are empty 


		if segment_area_thresh_low != 0 or segment_area_thresh_high != 1:
			various_sim_for_2models[f'greedyiou_thresh{segment_area_thresh_low}-{segment_area_thresh_high}'] = greedy_iou(sam_segments_filtered, attackers_segments_filtered)
			various_sim_for_2models[f'greedyiou_thresh{segment_area_thresh_low}-{segment_area_thresh_high}_nms'] = greedy_iou(sam_segments_filtered, attackers_segments_filtered_nms)
			various_sim_for_2models[f'greedyiou_thresh{segment_area_thresh_low}-{segment_area_thresh_high}_nms_crossReference'] = (greedy_iou(sam_segments_filtered, attackers_segments_filtered_nms) + greedy_iou(attackers_segments_filtered_nms, sam_segments_filtered)) / 2
			various_sim_for_2models[f'greedyiou_thresh{segment_area_thresh_low}-{segment_area_thresh_high}_nms_crossReference_areaWeighted'] = (greedy_iou(sam_segments_filtered, attackers_segments_filtered_nms, area_weighted=True) + greedy_iou(attackers_segments_filtered_nms, sam_segments_filtered, area_weighted=True)) / 2

		# various_sim_for_2models['greedyiou'] = greedy_iou(sam_segments, attackers_segments)
		# various_sim_for_2models['greedyiou_nms'] = greedy_iou(sam_segments, attackers_segments_nms)
		# various_sim_for_2models['greedyiou_nms_crossReference'] = (greedy_iou(sam_segments, attackers_segments_nms) + greedy_iou(attackers_segments_nms, sam_segments)) / 2
		# various_sim_for_2models['greedyiou_nms_crossReference_areaWeighted'] = (greedy_iou(sam_segments, attackers_segments_nms, area_weighted=True) + greedy_iou(attackers_segments_nms, sam_segments, area_weighted=True)) / 2
		various_sim_for_2models['greedyiou_crossReference'] = (greedy_iou(sam_segments, attackers_segments) + greedy_iou(attackers_segments, sam_segments)) / 2
			

	except Exception as error:
		print(error)
		# various_sim_for_2models['greedyiou'] = various_sim_for_2models['greedyiou_top5'] = \
		# various_sim_for_2models['greedyiou_top10'] = various_sim_for_2models['greedyiou_top15'] = \
		# various_sim_for_2models['greedyiou_top20'] = various_sim_for_2models['greedyiou_top30'] = \
		# various_sim_for_2models['greedyiou_area_weighted'] = -1

	
	data['sam'+'_'+'attackersEnsemble'].update(various_sim_for_2models)

	
	# Save the data
	print(data)
	with open(similarity_path+img_name+'.json', 'w') as file:
		json.dump(data, file)
	print(img_name, "saved")	
