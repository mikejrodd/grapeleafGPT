<p align="center" width="100%">
<img src="./images/grapeleaf.png" alt="AnomalyGPT_logo" style="width: 40%; min-width: 300px; display: block; margin: auto;" />
</p>

# grapeleafGPT: Detecting Esca Disease in Grape Vines using AnomalyGPT Large Vision-Language Model

![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-red.svg)

# GrapeLeafGPT: Anomaly Detection for Esca Disease in Grape Leaves

## Overview
GrapeLeafGPT is a machine learning project designed to detect Esca-infected grape leaves using an anomaly detection pipeline similar to AnomalyGPT, as described in the paper [AnomalyGPT](https://anomalygpt.github.io/). This project uses grape leaf images from the [Grape Disease Dataset on Kaggle](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original) and aims to identify Esca disease in grape leaves as anomalous while considering healthy leaves as normal.

## Background on Esca Disease
Esca disease is a complex and devastating grapevine trunk disease that affects vineyards worldwide. It manifests through various symptoms on grape leaves, including tiger stripe patterns, chlorosis, and necrosis. Infected vines suffer from reduced yield and grape quality, which severely impacts vineyard management and winemaking.

### Importance of Detecting Esca Disease
- **Yield Protection**: Early detection of Esca disease can help mitigate its spread, preserving the yield and quality of grape production.
- **Quality Control**: Ensuring the health of grapevines leads to higher quality grapes, essential for premium wine production.
- **Cost Efficiency**: Effective management and timely intervention can reduce the economic impact caused by the disease, saving costs on treatments and lost production.

## Models Used

### LLAMA (LLaMA: Large Language Model)
Developed by Meta, LLaMA is designed for a wide range of NLP tasks. It uses a transformer architecture and has been fine-tuned for specific tasks, including image understanding when combined with vision models. LLaMA is integrated with vision models to enhance its capabilities in visual tasks.

### PandaGPT
This model extends the capabilities of large pre-trained language models into the visual domain. PandaGPT aligns visual features with text features, enabling it to process and generate responses based on both textual and visual inputs. This alignment allows for effective performance in tasks like image captioning and visual question answering.

### AnomalyGPT
AnomalyGPT is a novel IAD approach based on LVLM. It eliminates the need for manual threshold adjustments by directly assessing the presence and location of anomalies. AnomalyGPT generates training data by simulating anomalous images and producing corresponding textual descriptions for each image. It employs an image decoder for fine-grained semantic understanding and a prompt learner to fine-tune the LVLM using prompt embeddings. AnomalyGPT supports multi-turn dialogues and exhibits impressive few-shot in-context learning capabilities.

### Key Components of AnomalyGPT

**Image Encoder**: Extracts features from input images using a pre-trained model like ImageBind.
```python
from transformers import ImageBindModel
image_encoder = ImageBindModel.from_pretrained("ImageBind-Huge")
```

**Text Encoder**: Converts textual descriptions and queries into embeddings using models like LLaMA and PandaGPT.
```python
from transformers import LlamaTokenizer, LlamaModel, PandaGPTTokenizer, PandaGPTModel
text_encoder = LlamaTokenizer.from_pretrained("Llama-7B")
text_model = LlamaModel.from_pretrained("Llama-7B")
```

**Prompt Learner**: Enhances the model with additional IAD knowledge using prompt embeddings.
```python
class PromptLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PromptLearner, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)
```

**Decoder**: Generates pixel-level anomaly localization results.
```python
class AnomalyDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AnomalyDecoder, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
```

**AnomalyGPT Integration**: Combines the components to detect and localize anomalies in industrial product images.
```python
class AnomalyGPT(nn.Module):
    def __init__(self, image_encoder, text_encoder, prompt_learner, decoder):
        super(AnomalyGPT, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.prompt_learner = prompt_learner
        self.decoder = decoder
    
    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        prompt_features = self.prompt_learner(text_features)
        anomaly_map = self.decoder(image_features + prompt_features)
        return anomaly_map
```

## test_grapeleaves.py performance:

- grapeleaves right: 1938 wrong: 1787
- grapeleaves i_AUROC: 62.49
- grapeleaves p_AUROC: 54.27
- i_AUROC: tensor(62.4900, dtype=torch.float64)
- p_AUROC: tensor(54.2700, dtype=torch.float64)
- precision: tensor(52.0268)

## Performance Assessment:

The performance of AnomalyGPT in detecting esca disease in grape leaves was suboptimal, as indicated by the metrics obtained: a precision of 52.03%, an i_AUROC of 62.49, and a p_AUROC of 54.27. These results suggest significant challenges in the model's ability to accurately identify and localize anomalies specific to esca.

A key aspect of AnomalyGPT's functioning is its reliance on precise localization of anomalies. The model generates pixel-level anomaly maps to highlight potential diseased areas in the leaves. These maps are then compared against ground truth masks to evaluate the model's performance. High localization accuracy is critical because any error in these pixel-level predictions can drastically affect the overall performance metrics, including precision and AUROC scores.

In this case, the ground truth masks were created through a largely manual verification process, which led to rudimentary and potentially inconsistent mask images. These masks are essential for training and evaluating the model, as they provide the reference for what constitutes an anomaly in the context of esca disease. However, the rudimentary nature of the masks means they may not capture all the nuances and variations in the symptoms of esca accurately. Esca disease symptoms can vary significantly in color, shape, and location on the leaf. This variability introduces additional complexity, as the model needs to generalize well across different manifestations of the disease. Additionally, leaves with esca may exhibit symptoms in the same locations as leaves with other diseases, leading to potential overlap. This overlap can confuse the model, causing it to misclassify or fail to detect esca-specific anomalies accurately.

The model's current architecture and training approach may not be well-suited to distinguish esca-specific anomalies from those of other diseases. Instead, AnomalyGPT appears to be more adept at broadly identifying "disease" rather than the specific characteristics of esca. This broad detection capability is reflected in the relatively low precision and AUROC scores, indicating that while the model can detect anomalies, it struggles to localize them accurately to esca-specific symptoms.

In short, the poor performance of AnomalyGPT in this task can be attributed to several factors:

- The model's heavy dependence on accurate pixel-level anomaly maps means that any errors here can significantly degrade performance.
- The manual and basic creation of ground truth masks likely introduced inconsistencies and inaccuracies.
- The variability in esca symptoms and their potential overlap with other diseases made it difficult for the model to accurately detect and localize esca-specific anomalies.
- AnomalyGPT's architecture may be more suited to detecting general disease symptoms rather than the specific indicators of esca, leading to lower precision and AUROC scores.

Improving the localization accuracy, refining the ground truth masks, and possibly modifying the model architecture to better capture the specific characteristics of esca could potentially enhance the model's performance in future iterations.





****

### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

<p align="center" width="100%">
<img src="./images/compare.png" alt="AnomalyGPT_logo" style="width: 80%; min-width: 400px; display: block; margin: auto;" />
</p>

**AnomalyGPT** is the first Large Vision-Language Model (LVLM) based Industrial Anomaly Detection (IAD) method that can detect anomalies in industrial images without the need for manually specified thresholds. Existing IAD methods can only provide anomaly scores and need manually threshold setting, while existing LVLMs cannot detect anomalies in the image. AnomalyGPT can not only indicate the presence and location of anomaly but also provide information about the image.

<img src="./images/AnomalyGPT.png" alt="AnomalyGPT" style="zoom:100%;" />

We leverage a pre-trained image encoder and a Large Language Model (LLM) to align IAD images and their corresponding textual descriptions via simulated anomaly data. We employ a lightweight, visual-textual feature-matching-based image decoder to obtain localization result, and design a prompt learner to provide fine-grained semantic to LLM and fine-tune the LVLM using prompt embeddings. Our method can also detect anomalies for previously unseen items with few normal sample provided.  


****


### Citation:

If you found AnomalyGPT useful in your research or applications, please kindly cite using the following BibTeX:
```
@article{gu2023anomalyagpt,
  title={AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models},
  author={Gu, Zhaopeng and Zhu, Bingke and Zhu, Guibo and Chen, Yingying and Tang, Ming and Wang, Jinqiao},
  journal={arXiv preprint arXiv:2308.15366},
  year={2023}
}
```


****

<span id='acknowledgments'/>

### Acknowledgments:

We borrow some codes and the pre-trained weights from [PandaGPT](https://github.com/yxuansu/PandaGPT). Thanks for their wonderful work!


[![Star History Chart](https://api.star-history.com/svg?repos=CASIA-IVA-Lab/AnomalyGPT&type=Date)](https://star-history.com/#CASIA-IVA-Lab/AnomalyGPT&Date)


 
