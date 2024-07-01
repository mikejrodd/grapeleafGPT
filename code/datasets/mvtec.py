import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .self_sup_tasks import patch_ex

WIDTH_BOUNDS_PCT = {
    'grapeleaves': ((0.03, 0.4), (0.03, 0.4))
}

NUM_PATCHES = {
    'grapeleaves': 4
}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {
    'grapeleaves': (1/12, 24)
}

# bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['grapeleaves']

# brightness, threshold pairs
BACKGROUND = {
    'grapeleaves': (200, 60)
}

OBJECTS = [
    'grapeleaves'
]

TEXTURES = []

describles = {}
describles['grapeleaves'] = "This is a photo of healthy grape leaves for anomaly detection, which should be green, without any damage, flaw, defect, scratch, hole or brown part."


# WIDTH_BOUNDS_PCT = {
#     'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)), 
#     'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)), 
#     'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)), 
#     'zipper':((0.03, 0.4), (0.03, 0.2)), 'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)), 
#     'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4)),
#     'grapeleaves':((0.03, 0.4), (0.03, 0.4))
# }

# NUM_PATCHES = {
#     'bottle':3, 'cable':3, 'capsule':3, 'hazelnut':3, 'metal_nut':3, 'pill':3, 
#     'screw':4, 'toothbrush':3, 'transistor':3, 'zipper':4, 'carpet':4, 'grid':4, 
#     'leather':4, 'tile':4, 'wood':4, 'grapeleaves':4
# }

# # k, x0 pairs
# INTENSITY_LOGISTIC_PARAMS = {
#     'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7), 
#     'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
#     'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15), 'grapeleaves':(1/12, 24)
# }

# # bottle is aligned but it's symmetric under rotation
# UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw', 'grapeleaves']

# # brightness, threshold pairs
# BACKGROUND = {
#     'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60), 
#     'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20), 'grapeleaves': (200, 60)
# }

# OBJECTS = [
#     'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 
#     'toothbrush', 'transistor', 'zipper', 'grapeleaves'
# ]
# TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']

# describles = {}
# describles['bottle'] = "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
# describles['cable'] = "This is a photo of three cables for anomaly detection, they are green, blue and grey, which cannot be missed or swapped and should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['capsule'] = "This is a photo of a capsule for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part."
# describles['carpet'] = "This is a photo of carpet for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['grid'] = "This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['hazelnut'] = "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['leather'] = "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
# describles['metal_nut'] = "This is a photo of a metal nut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped."
# describles['pill'] = "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
# describles['screw'] = "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
# describles['tile'] = "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['toothbrush'] = "This is a photo of a toothbrush for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['transistor'] = "This is a photo of a transistor for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['wood'] = "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
# describles['zipper'] = "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['grapeleaves'] = "This is a photo of healthy grape leaves for anomaly detection, which should be green, without any damage, flaw, defect, scratch, hole or brown part."

class MVtecDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.paths = []
        self.x = []
        self.masks = []  # Define the self.masks attribute
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        train_good_count = 0  # Counter for train/good images
        ground_truth_count = 0  # Counter for ground_truth images

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train/good" in file_path and file.lower().endswith(valid_extensions):
                    self.paths.append(file_path)
                    try:
                        self.x.append(self.transform(Image.open(file_path).convert('RGB')))
                        train_good_count += 1  # Increment the counter for each successful image load
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        continue
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
                        continue
                elif "ground_truth" in file_path and file.lower().endswith(valid_extensions):
                    try:
                        mask = self.transform(Image.open(file_path).convert('L'))  # Assuming masks are grayscale
                        self.masks.append(mask)  # Append the transformed mask
                        ground_truth_count += 1  # Increment the counter for each successful image load
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        continue
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
                        continue

        self.prev_idx = np.random.randint(len(self.paths))

        # Debug statement to check if paths and masks are loaded
        if train_good_count == 0:
            print(f"No train/good data found in {self.root_dir}")
        else:
            print(f"Loaded {train_good_count} images from train/good in {self.root_dir}")

        if ground_truth_count == 0:
            print(f"No ground_truth data found in {self.root_dir}")
        else:
            print(f"Loaded {ground_truth_count} images from ground_truth in {self.root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        try:
            x = self.x[index]
        except (FileNotFoundError, IndexError):
            print(f"File not found or index error: {img_path}")
            return None

        # Manually set the class name to 'grapeleaves'
        class_name = 'grapeleaves'

        # Debug statement to check class_name
        # print(f"class_name: {class_name}")

        self_sup_args = {
            'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
            'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name),
            'num_patches': 2,
            'min_object_pct': 0,
            'min_overlap_pct': 0.25,
            'gamma_params': (2, 0.05, 0.03), 
            'resize': True, 
            'shift': True, 
            'same': False, 
            'mode': cv2.NORMAL_CLONE,
            'label_mode': 'logistic-intensity',
            'skip_background': BACKGROUND.get(class_name)
        }

        # Debug statement to check self_sup_args
        # print(f"self_sup_args: {self_sup_args}")

        if self_sup_args['width_bounds_pct'] is None:
            raise ValueError(f"width_bounds_pct is None for class {class_name}")

        x = np.asarray(x)
        origin = x

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)    
        x, mask, centers = patch_ex(x, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index

        origin = self.norm_transform(origin)
        x = self.norm_transform(x)

        if len(centers) > 0:
            position = []
            for center in centers:
                center_x = center[0] / 224
                center_y = center[1] / 224

                if center_x <= 1/3 and center_y <= 1/3:
                    position.append('top left')
                elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('top')
                elif center_x <= 1/3 and center_y > 2/3:
                    position.append('top right')

                elif center_x <= 2/3 and center_y <= 1/3:
                    position.append('left')
                elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('center')
                elif center_x <= 2/3 and center_y > 2/3:
                    position.append('right')

                elif center_y <= 1/3:
                    position.append('bottom left')
                elif center_y > 1/3 and center_y <= 2/3:
                    position.append('bottom')
                elif center_y > 2/3:
                    position.append('bottom right')

            conversation_normal = []
            conversation_normal.append({"from": "human", "value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from": "gpt", "value": "No, there is no anomaly in the image."})

            conversation_abnormal = []
            conversation_abnormal.append({"from": "human", "value": describles[class_name] + " Is there any anomaly in the image?"})

            if len(centers) > 1:
                abnormal_describe = "Yes, there are " + str(len(centers)) + " anomalies in the image, they are at the "
                for i in range(len(centers)):
                    if i == 0:
                        abnormal_describe += position[i]
                    elif i == 1 and position[i] != position[i - 1]:
                        if i != len(centers) - 1:
                            abnormal_describe += ", "
                            abnormal_describe += position[i]
                        else:
                            abnormal_describe += " and " + position[i] + " of the image."
                    elif i == 1 and position[i] == position[i - 1]:
                        if i == len(centers) - 1:
                            abnormal_describe += " of the image."
            else:
                abnormal_describe = "Yes, there is an anomaly in the image, at the " + position[0] + " of the image."

            conversation_abnormal.append({"from": "gpt", "value": abnormal_describe})

        else:
            print("No mask")
            conversation_normal = []
            conversation_normal.append({"from": "human", "value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from": "gpt", "value": "No, there is no anomaly in the image."})

            conversation_abnormal = conversation_normal

        return origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path

    def collate(self, instances):
        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []

        for i, instance in enumerate(instances):
            if instance is None:
                continue

            # Add the first set of data
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[4])
            masks.append(torch.zeros_like(instance[5]))
            img_paths.append(instance[6])

            # Add the second set of data
            images.append(instance[2])
            texts.append(instance[3])
            class_names.append(instance[4])
            masks.append(instance[5])
            img_paths.append(instance[6])

        # # Print the summary of the batch
        # print(f'\nSummary of processed instances:')
        # print(f'  Total instances processed: {len(instances)}')
        # print(f'  Total images collected: {len(images)}')
        # print(f'  Total texts collected: {len(texts)}')
        # print(f'  Total class names collected: {len(class_names)}')
        # print(f'  Total masks collected: {len(masks)}')
        # print(f'  Total img paths collected: {len(img_paths)}')
        # print(f'  Img Paths Details: {img_paths}\n')

        return {
            'images': images,
            'texts': texts,
            'class_names': class_names,
            'masks': masks,
            'img_paths': img_paths
        }
