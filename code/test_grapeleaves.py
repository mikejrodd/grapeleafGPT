import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
from model.openllama import OpenLLAMAPEFTModel

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)

command_args = parser.parse_args()

describles = {}
describles['grapeleaves'] = "This is a photo of a grape leaf for anomaly detection. The leaf should be green and healthy, without any signs of esca disease, which includes discoloration, brown spots, or unusual texture. The leaf may appear damaged or diseased in a way not related to esca infection."

FEW_SHOT = command_args.few_shot 

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_grapeleaves/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

"""Override Chatbot.postprocess"""
p_auc_list = []
i_auc_list = []

def predict(
    input, 
    image_paths, 
    normal_img_paths, 
    max_length, 
    top_p, 
    temperature,
    history,
    modality_cache,  
):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': image_paths if image_paths else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_paths if normal_img_paths else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return response, pixel_output

class GrapeLeavesTestDataset(Dataset):
    def __init__(self, root_dir, describles, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.describles = describles
        self.data = []

        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

        for root, _, files in os.walk(root_dir):
            for file in files:
                if "test" in root and file.lower().endswith(valid_extensions) and ('esca' in root or 'good' in root):
                    file_path = os.path.join(root, file)
                    self.data.append((file_path, describles['grapeleaves']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, description = self.data[idx]
        is_normal = 'good' in file_path.split('/')[-2]

        if is_normal:
            img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
        else:
            mask_path = file_path.replace('test', 'ground_truth')
            mask_path = mask_path.replace('.png', '_mask.png').replace('.jpg', '_mask.jpg').replace('.jpeg', '_mask.jpeg')
            mask_path = mask_path.replace('.PNG', '_mask.PNG').replace('.JPG', '_mask.JPG').replace('.JPEG', '_mask.JPEG')
            if not os.path.exists(mask_path):
                print(f'Mask not found for {file_path}. Skipping...')
                return None, None, None, None
            img_mask = Image.open(mask_path).convert('L')

        img_mask = self.transform(img_mask)
        img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
        img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
        
        return file_path, description, img_mask, is_normal

input = "Is there any anomaly in the image?"
root_dir = '../data/grapeleaves'

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES = ['grapeleaves']

precision = []

for c_name in CLASS_NAMES:
    train_dir = os.path.join(root_dir, "train", "good")
    if not os.path.exists(train_dir):
        print(f'Training directory does not exist: {train_dir}')
        continue
    
    normal_img_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"))][:command_args.k_shot]

    # Debugging: print the normal image paths
    print(f'Normal image paths for {c_name}:', normal_img_paths)
    if not normal_img_paths:
        print(f'No images found in training directory: {train_dir}')
        continue
    
    test_dataset = GrapeLeavesTestDataset(root_dir, describles, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    
    for batch in test_loader:
        batch = [item for item in batch if item[0] is not None]  # Filter out None items
        if not batch:
            continue
        
        file_paths, descriptions, img_masks, is_normals = zip(*batch)

        for file_path, description, img_mask, is_normal in zip(file_paths, descriptions, img_masks, is_normals):
            if FEW_SHOT:
                resp, anomaly_map = predict(description + ' ' + input, [file_path], normal_img_paths, 512, 0.1, 1.0, [], [])
            else:
                resp, anomaly_map = predict(description + ' ' + input, [file_path], [], 512, 0.1, 1.0, [], [])
            
            img_mask = np.array(img_mask)
            anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

            p_label.append(img_mask)
            p_pred.append(anomaly_map)

            i_label.append(1 if not is_normal else 0)
            i_pred.append(anomaly_map.max())

            if 'good' not in file_path and 'Yes' in resp:
                right += 1
            elif 'good' in file_path and 'No' in resp:
                right += 1
            else:
                wrong += 1

    if p_pred and p_label:  # Check if there are any predictions and labels
        p_pred = np.array(p_pred)
        p_label = np.array(p_label)

        i_pred = np.array(i_pred)
        i_label = np.array(i_label)

        p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
        i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    
        p_auc_list.append(p_auroc)
        i_auc_list.append(i_auroc)
        precision.append(100 * right / (right + wrong))

        print(c_name, 'right:', right, 'wrong:', wrong)
        print(c_name, "i_AUROC:", i_auroc)
        print(c_name, "p_AUROC:", p_auroc)
    else:
        print(f'No predictions or labels found for {c_name}. Skipping AUROC calculation.')

if i_auc_list and p_auc_list:  # Check if there are any AUROC scores calculated
    print("i_AUROC:", torch.tensor(i_auc_list).mean())
    print("p_AUROC:", torch.tensor(p_auc_list).mean())
    print("precision:", torch.tensor(precision).mean())
else:
    print("No AUROC scores calculated. Please check your dataset and paths.")
