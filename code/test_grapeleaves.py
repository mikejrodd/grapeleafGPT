import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast

class GrapeLeavesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if "test" in root and file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.file_paths.append(os.path.join(root, file))
                    if 'esca' in root:
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return img_path, image, label

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=5)  # Set k_shot to 5
parser.add_argument("--round", type=int, default=3)  # Number of epochs
parser.add_argument("--batch_size", type=int, default=32)  # Increase batch size
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

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

root_dir = '../data/grapeleaves'  # Ensure root_dir is defined

dataset = GrapeLeavesDataset(root_dir=root_dir, transform=test_transform)

# Create a balanced sampler
class_count = [sum(dataset.labels), len(dataset.labels) - sum(dataset.labels)]
weights = 1. / torch.tensor(class_count, dtype=torch.float)
samples_weights = weights[dataset.labels]
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

dataloader = DataLoader(dataset, batch_size=command_args.batch_size, sampler=sampler, num_workers=4)  # Increase num_workers

CLASS_NAMES = ['grapeleaves']  # Move this to the correct scope

def log_metrics(epoch, batch_idx, right, wrong, p_auc_list, i_auc_list):
    precision = 100 * right / (right + wrong) if (right + wrong) > 0 else 0
    p_auroc = np.mean(p_auc_list) if p_auc_list else 0
    i_auroc = np.mean(i_auc_list) if i_auc_list else 0
    print(f'Epoch: {epoch}, Batch: {batch_idx}, Precision: {precision:.2f}, p_AUROC: {p_auroc:.2f}, i_AUROC: {i_auroc:.2f}')

def predict(
    input, 
    image_path, 
    normal_img_path, 
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

    with autocast():
        response, pixel_output = model.generate({
            'prompt': prompt_text,
            'image_paths': [image_path] if image_path else [],
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'normal_img_paths': normal_img_path if normal_img_path else [],
            'top_p': top_p,
            'temperature': temperature,
            'max_tgt_len': max_length,
            'modality_embeds': modality_cache
        })

    return response, pixel_output

input = "Is there any anomaly in the image?"

# Initialize metrics lists
p_auc_list = []
i_auc_list = []
precision_list = []

for epoch in range(command_args.round):
    for c_name in CLASS_NAMES:
        train_dir = os.path.join(root_dir, "train", "good")
        if not os.path.exists(train_dir):
            print(f'Training directory does not exist: {train_dir}')
            continue
        
        normal_img_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:command_args.k_shot]

        # Debugging: print the normal image paths
        print(f'Normal image paths for {c_name}:', normal_img_paths)
        if not normal_img_paths:
            print(f'No images found in training directory: {train_dir}')
        
        right = 0
        wrong = 0
        p_pred = []
        p_label = []
        i_pred = []
        i_label = []

        for batch_idx, batch in enumerate(dataloader):
            img_paths, images, labels = batch
            
            batch_p_preds = []
            batch_p_labels = []
            batch_i_preds = []
            batch_i_labels = []

            for img_path, image, label in zip(img_paths, images, labels):
                if FEW_SHOT:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, img_path, normal_img_paths, 512, 0.1, 1.0, [], [])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, img_path, [], 512, 0.1, 1.0, [], [])
                
                if label == 0:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = img_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    mask_path = mask_path.replace('.jpg', '_mask.png')
                    mask_path = mask_path.replace('.jpeg', '_mask.png')
                    if not os.path.exists(mask_path):
                        mask_path = img_path.replace('.png', '_mask.jpg')
                        mask_path = img_path.replace('.jpg', '_mask.jpg')
                        mask_path = img_path.replace('.jpeg', '_mask.jpg')
                    if not os.path.exists(mask_path):
                        print(f'Mask not found for {img_path}. Skipping...')
                        continue
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                
                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

                batch_p_labels.append(img_mask)
                batch_p_preds.append(anomaly_map)

                batch_i_labels.append(label.item())  # Ensure label is converted to a scalar
                batch_i_preds.append(anomaly_map.max().item())  # Ensure anomaly_map.max() is a scalar

                if label == 1 and 'Yes' in resp:
                    right += 1
                elif label == 0 and 'No' in resp:
                    right += 1
                else:
                    wrong += 1

            p_label.extend(batch_p_labels)
            p_pred.extend(batch_p_preds)
            i_label.extend(batch_i_labels)
            i_pred.extend(batch_i_preds)

            if batch_idx % 2 == 0:
                if p_label and p_pred:  # Check if lists are not empty
                    p_auroc = round(roc_auc_score(np.array(p_label).ravel(), np.array(p_pred).ravel()) * 100, 2)
                else:
                    p_auroc = 0

                if len(set(i_label)) > 1:  # Check if there are both classes in i_label
                    i_auroc = round(roc_auc_score(np.array(i_label), np.array(i_pred)) * 100, 2)
                else:
                    i_auroc = 0

                p_auc_list.append(p_auroc)
                i_auc_list.append(i_auroc)
                log_metrics(epoch, batch_idx, right, wrong, p_auc_list, i_auc_list)
                print(f'Batch index: {batch_idx}, right: {right}, wrong: {wrong}, p_auroc: {p_auroc}, i_auroc: {i_auroc}')  # Debugging print
                print(f'i_label: {i_label}, i_pred: {i_pred}')  # Log i_label and i_pred

        if p_pred and p_label:  # Check if there are any predictions and labels
            p_pred = np.array(p_pred)
            p_label = np.array(p_label)

            i_pred = np.array(i_pred)
            i_label = np.array(i_label)

            p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
            if len(set(i_label)) > 1:  # Check if there are both classes in i_label
                i_auroc = round(roc_auc_score(i_label, i_pred) * 100, 2)
            else:
                i_auroc = 0
        
            p_auc_list.append(p_auroc)
            i_auc_list.append(i_auroc)
            precision_list.append(100 * right / (right + wrong))

            print(c_name, 'right:', right, 'wrong:', wrong)
            print(c_name, "i_AUROC:", i_auroc)
            print(c_name, "p_AUROC:", p_auroc)
        else:
            print(f'No predictions or labels found for {c_name}. Skipping AUROC calculation.')

if i_auc_list and p_auc_list:  # Check if there are any AUROC scores calculated
    print("i_AUROC:", torch.tensor(i_auc_list).mean().item())
    print("p_AUROC:", torch.tensor(p_auc_list).mean().item())
    print("precision:", torch.tensor(precision_list).mean().item())
else:
    print("No AUROC scores calculated. Please check your dataset and paths.")
