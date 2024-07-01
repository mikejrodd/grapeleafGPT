import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)

command_args = parser.parse_args()

describles = {}
describles['bottle'] = "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['cable'] = "This is a photo of three cables for anomaly detection, cables cannot be missed or swapped, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['capsule'] = "This is a photo of a capsule for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part."
describles['carpet'] = "This is a photo of carpet for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['grid'] = "This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['hazelnut'] = "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['leather'] = "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
describles['metal_nut'] = "This is a photo of a metal nut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped."
describles['pill'] = "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['screw'] = "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
describles['tile'] = "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['toothbrush'] = "This is a photo of a toothbrush for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['transistor'] = "This is a photo of a transistor for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['wood'] = "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['zipper'] = "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['grapeleaves'] = "This is a photo of a grape leaf for anomaly detection. The leaf should be green and healthy, without any signs of esca disease, which includes discoloration, brown spots, or unusual texture. The leaf may appear damaged or diseased in a way not related to esca infection."

FEW_SHOT = command_args.few_shot

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt',
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
root_dir = '../data/mvtec_anomaly_detection'

mask_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])

CLASS_NAMES = ['grapeleaves','bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 'transistor', 'wood', 'zipper']

precision = []

for c_name in CLASS_NAMES:
# Dynamically list all image files in the directory
    all_images = sorted([os.path.join("../data/mvtec_anomaly_detection/", c_name, "train", "good", f) 
                        for f in os.listdir(f"../data/mvtec_anomaly_detection/{c_name}/train/good/") 
                        if f.endswith('.png')])

    # Select the necessary number of samples based on k_shot
    normal_img_paths = all_images[:command_args.k_shot]

    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and file.split('.')[-1].lower() in ['png', 'jpeg', 'jpg'] and c_name in file_path:
                if FEW_SHOT:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, normal_img_paths, 512, 0.1, 1.0, [], [])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, [], 512, 0.1, 1.0, [], [])
                is_normal = 'good' in file_path.split('/')[-2]

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth').replace('/esca', '')
                    mask_path = os.path.splitext(mask_path)[0] + '_mask.jpg'
                    try:
                        img_mask = Image.open(mask_path).convert('L')
                    except FileNotFoundError:
                        print(f"Mask not found for image {file_path}. Skipping.")
                        continue


                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                
                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

                p_label.append(img_mask)
                p_pred.append(anomaly_map)

                i_label.append(1 if not is_normal else 0)
                i_pred.append(anomaly_map.max())

                position = []

                if 'good' not in file_path and 'Yes' in resp:
                    right += 1
                elif 'good' in file_path and 'No' in resp:
                    right += 1
                else:
                    wrong += 1

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100,2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100,2)
    
    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * right / (right + wrong))

    print(c_name, 'right:',right,'wrong:',wrong)
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

print("i_AUROC:",torch.tensor(i_auc_list).mean())
print("p_AUROC:",torch.tensor(p_auc_list).mean())
print("precision:",torch.tensor(precision).mean())

