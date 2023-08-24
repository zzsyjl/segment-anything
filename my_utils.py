import glob
import os
import io
from collections import defaultdict
import zlib
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import clip



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def non_max_suppression(masks, iou_threshold=0.7):
    # Sort masks based on area in descending order
    sorted_masks = sorted(masks, key=lambda mask: mask.sum(), reverse=True)
    selected_indices = []
    selected_masks = []

    for i, mask in enumerate(sorted_masks):
        is_suppressed = False
        
        for selected_idx in selected_indices:
            selected_mask = sorted_masks[selected_idx]
            iou = calculate_iou_mask(mask, selected_mask)
            
            if iou > iou_threshold:
                is_suppressed = True
                break
        
        if not is_suppressed:
            selected_indices.append(i)
            selected_masks.append(mask)
    return selected_masks

def calculate_iou_mask(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0.0
    return iou

def delete_png_files():
    # delete png files reccursively from cur dir
    for f in glob('./**/*.png', recursive=True):
        print(f)
        os.remove(f)


def resize_images(tar_dir, tar_size, ori_dir='.'):
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    for f in glob(ori_dir + '/**/*.jpg', recursive=True):
        print(f)
        img = cv2.imread(f)
        height, width, channel = img.shape
        if height > width:
            img = cv2.resize(img, (tar_size, int(tar_size * height / width)))
        else:
            img = cv2.resize(img, (int(tar_size * width / height), tar_size))
        cv2.imwrite(tar_dir + '/' + f.split('/')[-1], img)


def clip_and_sam(img, text):
    tokenized_text = clip.tokenize([text]).to(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_features = model.encode_text(tokenized_text)

def show_img_and_attacker_segment_info(img_name='000000374727', models=['oneformer', 'mask2former', 'maskformer']):
	# show image
	img = Image.open(f'/home/yjl/data/val2017/{img_name}.jpg')
	plt.imshow(img)
	print('img.size:', img.size)
	# show saved results
	for m in models:
		segmentation, segments_info = load_segmentation_and_info(f'/home/yjl/data/val2017_{m}_results/{img_name}')
		print('-'*30, m, '-'*30)
		print(segmentation.shape)
		print(torch.unique(segmentation))
		print(segments_info)

def greedy_iou(segmentation1, segmentation2, top_n=False, area_weighted=False):
    """
    Compute the IoU between two segmentations using a greedy algorithm.
    SAM segments is accustomed as argument segmentation1.
    If top_n is not False, only the top_n segments in segmentation1 are considered.
    datatype: torch.tensor, not numpy.ndarray
    """
    if isinstance(segmentation1, list):
        n_segments1 = len(segmentation1)
    else:
        n_segments1 = torch.max(segmentation1)
    if isinstance(segmentation2, list):
        n_segments2 = len(segmentation2)
    else:
        n_segments2 = torch.max(segmentation2)
    # initialize the IoU matrix
    ious = torch.zeros((n_segments1, n_segments2))
    for segment1_id in range(1, n_segments1 + 1):
        for segment2_id in range(1, n_segments2 + 1):
            if isinstance(segmentation1, list):
                mask1 = segmentation1[segment1_id - 1]
            else:
                mask1 = segmentation1 == segment1_id
            if isinstance(segmentation2, list):
                mask2 = segmentation2[segment2_id - 1]
            else:
                mask2 = segmentation2 == segment2_id
            # compute the IoU
            ious[segment1_id - 1, segment2_id - 1] = (mask1 & mask2).sum() / (mask1 | mask2).sum()
    # get the IoU for each segment in the first segmentation
    ious = ious.max(dim=1)[0]
    if area_weighted:
        # get the area of each segment in the first segmentation
        areas = torch.zeros(n_segments1)
        for segment1_id in range(1, n_segments1 + 1):
            if isinstance(segmentation1, list):
                mask1 = segmentation1[segment1_id - 1]
            else:
                mask1 = segmentation1 == segment1_id
            areas[segment1_id - 1] = mask1.sum()
        ious = ious * areas / (mask1.shape[0] * mask1.shape[1]) * len(areas)
    if top_n and top_n < n_segments1:
        # get the top_n iou values
        ious, _ = ious.topk(top_n)
    return ious.mean().item()


def save_segmentation_and_info(segmentation, segments_info, save_path):
    """
    segmentation: torch.tensor of int32(e.g. 0,1,2,3,4,5), shape=(H, W)
    segments_info: list
    save_path: str

    initialized for mask2former, hope to be generalized
    """
    # save segmentation with zlib
    if isinstance(segmentation, bytes):  # detr results
        compressed_data = zlib.compress(segmentation)
    else:
        compressed_data = zlib.compress(segmentation.numpy().tobytes())
    with open(save_path + '.zlib', 'wb') as f:
        f.write(compressed_data)

    # save shape and segments_info with pickle
    if isinstance(segmentation, bytes):  # detr results
        metadata = {'segments_info': segments_info}
    else:
        shape = segmentation.shape
        metadata = {'shape': shape, 'segments_info': segments_info}
    with open(save_path + '.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_segmentation_and_info(load_path='/home/yjl/data/val2017_oneformer_results/000000000139', return_list=False):
    """
    load_path: str. Will add .zlib and .pkl automatically

    return:
            segmentation: torch.tensor of int32(e.g. 0,1,2,3,4,5), shape=(H, W)
            segments_info: list of segment info, e.g. {'id': 1, 'label_id': 75, 'was_fused': False, 'score': 0.827459}
    """
    # load segmentation tensor with zlib
    with open(load_path + '.zlib', 'rb') as f:
        compressed_data = f.read()
    # load metadata with pickle
    with open(load_path + '.pkl', 'rb') as f:
        metadata = pickle.load(f)

    if 'detr' in load_path:  # detr saves png string as bytes
        print(compressed_data)
        panoptic_seg = Image.open(io.BytesIO(compressed_data))
        panoptic_seg = torch.from_numpy(np.array(panoptic_seg, dtype=np.uint8))
    else:
        segmentation = np.frombuffer(zlib.decompress(compressed_data), dtype=np.int32)
        segmentation = torch.from_numpy(segmentation)
        shape = metadata['shape']
        segmentation = segmentation.reshape(shape)
        # maskformer saves smaller tensor, resize to original size as oneformer result
        if 'maskformer' in load_path:
            oneformer_metadata = pickle.load(
                open(load_path.replace('maskformer', 'oneformer') + '.pkl', 'rb'))
            original_shape = oneformer_metadata['shape']
            segmentation = torch.nn.functional.interpolate(segmentation.unsqueeze(0).unsqueeze(
                0).float(), size=original_shape, mode='nearest').squeeze(0).squeeze(0).long()
    # make a list of binary masks out of segmentation
    if return_list:
        segmentation = [segmentation == segment_id for segment_id in range(1, torch.max(segmentation) + 1)]

    segments_info = metadata['segments_info']
    return segmentation, segments_info


def save_model_results_for_imgs(model_name, img_folder='/home/yjl/data/val2017'):

    if not os.path.exists(f'/home/yjl/data/val2017_{model_name}_results'):
        os.makedirs(f'/home/yjl/data/val2017_{model_name}_results')
    img_paths = sorted(glob.glob(img_folder + '/*'))
    for img_path in img_paths:
        # get the name of the image
        image = Image.open(img_path)
        img_name = img_path.split('/')[-1].split('.')[0]
        print(img_name)
        try:
            if model_name == 'oneformer':
                from transformers import AutoProcessor, AutoModelForUniversalSegmentation
                processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
                model = AutoModelForUniversalSegmentation.from_pretrained(
                    "shi-labs/oneformer_coco_swin_large")
                panoptic_inputs = processor(images=image, task_inputs=[
                                            "panoptic"], return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**panoptic_inputs)
                panoptic_segmentation = processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[image.size[::-1]])[0]
                save_segmentation_and_info(
                    **panoptic_segmentation, save_path=f"/home/yjl/data/val2017_oneformer_results/{img_name}")

            if model_name == 'mask2former':
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
                processor = AutoImageProcessor.from_pretrained(
                    "facebook/mask2former-swin-base-coco-panoptic")
                model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    "facebook/mask2former-swin-base-coco-panoptic")
                panoptic_inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**panoptic_inputs)
                panoptic_segmentation = processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[image.size[::-1]])[0]
                save_segmentation_and_info(
                    **panoptic_segmentation, save_path=f"/home/yjl/data/val2017_mask2former_results/{img_name}")

            if model_name == 'maskformer':
                from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
                processor = MaskFormerImageProcessor.from_pretrained(
                    "facebook/maskformer-swin-small-coco")
                model = MaskFormerForInstanceSegmentation.from_pretrained(
                    "facebook/maskformer-swin-small-coco")
                panoptic_inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**panoptic_inputs)
                panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs)[0]
                save_segmentation_and_info(
                    **panoptic_segmentation, save_path=f"/home/yjl/data/val2017_maskformer_results/{img_name}")

            if model_name == 'detr':
                from transformers import DetrFeatureExtractor, DetrForSegmentation
                feature_extractor = DetrFeatureExtractor.from_pretrained(
                    "facebook/detr-resnet-50-panoptic")
                model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
                encoding = feature_extractor(image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**encoding)
                processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
                result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
                save_segmentation_and_info(
                    result['png_string'], result['segments_info'], save_path=f"/home/yjl/data/val2017_detr_results/{img_name}")
        except:
            continue


def show_segmentation(segmentation, segments_info, img_path):
    """
    segmentation: torch.tensor of int32(e.g. 1,2,3,4,5), shape=(H, W)
    segments_info: list
    img_path: str
    """
    # load image
    image = Image.open(img_path)
    # get the name of the image
    img_name = img_path.split('/')[-1].split('.')[0]
    # get the number of segments
    n_segments = torch.max(segmentation)
    # initialize the colors
    colors = torch.rand((n_segments + 1, 3))
    # set the color of the background to black
    colors[0] = 0
    # set the color of the segments
    for segment_id in range(1, n_segments + 1):
        segments_info[segment_id - 1]['color'] = colors[segment_id].tolist()
    # create the segmentation mask
    segmentation_mask = torch.zeros((segmentation.shape[0], segmentation.shape[1], 3))
    for segment_id in range(1, n_segments + 1):
        segmentation_mask[segmentation == segment_id] = torch.tensor(
            segments_info[segment_id - 1]['color'])
    # show the segmentation mask
    plt.imshow(segmentation_mask)
    plt.show()
    
def draw_panoptic_segmentation(segmentation, segments_info, id2label, ax=None, legend=True):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
	# for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        # get the id2label mapping from mask2former
        
        segment_label = id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
    if legend:
        ax.legend(handles=handles)
        
def show_segmentation(ax=None, img_name='000000000139', model_name='oneformer', mask_generator=None, transparency=0.5, id2label=None, legend=True):
    """
    compatible with SAM and other attackers
    img_path: str
    """

    if model_name == 'sam':
        image = cv2.imread(f'/home/yjl/data/val2017/{img_name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        ax.imshow(image)
        show_anns(masks, ax, transparency)
    else:
        # load segmentation and segments_info
        segmentation, segments_info = load_segmentation_and_info(f'/home/yjl/data/val2017_{model_name}_results/{img_name}')
        draw_panoptic_segmentation(segmentation, segments_info, id2label, ax, legend)
    # turn off the axis
    ax.axis('off')
    
def show_anns(anns, ax=None, transparency=0.5):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
        print(ax)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1-transparency]])
        img[m] = color_mask
    ax.imshow(img)

def sam_demo(img_path, mask_generator, transparency=0.5):
    """ Given img path, over lay the segmentation on the image.
        Model has been loaded before calling this function.
    """
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    # plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks, ax=None, transparency=transparency)
    plt.show() 



	

if __name__ == '__main__':
    # delete_png_files()
    # save_model_results_for_imgs('detr')
	# show_segmentation()
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_demo("/home/yjl/data/val2017/000000000139.jpg", mask_generator, transparency=0.5)


