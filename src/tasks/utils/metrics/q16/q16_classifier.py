import torch
import PIL
import pickle
import clip
# from transformers import CLIPProcessor,CLIPModel, AutoTokenizer,AutoProcessor
## get text_features
r"""
>>> from transformers import AutoTokenizer, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```"""
## get iamge_features
r"""
Returns:
    image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
    applying the projection layer to the pooled output of [`CLIPVisionModel`].

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```"""
class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14'):
        super(ClipWrapper, self).__init__()
        self.clip_model, self.preprocess = clip.load(model_name,device,jit=False,download_root='.cache')
        # self.clip_model, self.preprocess = clip.load(model_name,device,jit=False)
        # self.model = CLIPModel.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
        # self.processor = AutoProcessor.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
        # self.tokenizer = AutoTokenizer.from_pretrained("/home/majc/openai/clip-vit-large-patch14")
        self.clip_model.eval()

    def forward(self, x):
        # inputs = self.processor(images=x, return_tensors="pt")
        # return self.model.get_image_features(**inputs)
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings).to(device)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1,
                                                                 keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features_norm @ embeddings_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def initialize_prompts(clip_model, text_prompts, device):
    text = clip.tokenize(text_prompts).to(device)
    # text = clip.tokenizer(text_prompts, padding=True, return_tensors="pt")
    # return clip.model.get_text_features(**text)
    return clip_model.encode_text(text)


def save_prompts(classifier, save_path):
    prompts = classifier.embeddings.detach().cpu().numpy()
    pickle.dump(prompts, open(save_path, 'wb'))


def load_prompts(file_path, device):
    return torch.HalfTensor(pickle.load(open(file_path, 'rb')))

def compute_embeddings(clip_model, image, device):
    image = PIL.Image.open(image)
    images = [clip_model.preprocess(image)]
    images = torch.stack(images).to(device)
    return clip_model(images).half()
    # images = clip.processor(images=image, return_tensors="pt").to(device)
    # images = torch.stack(images).to(device)
    # return clip.model.get_image_features(**images).half()