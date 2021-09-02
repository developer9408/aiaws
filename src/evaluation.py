import torch
from albumentations import (Compose, Normalize, Resize)
from .utils import CLASS_TO_LABEL, load_checkpoint
from .model import NutriNet

def preprocess_rgb_input(image) -> torch.FloatTensor:
    basic_aug = Compose([
        Normalize(),
        Resize(224, 224),
        #                 PadIfNeeded(min_height=224, min_width=224)
    ])
    im = basic_aug(image=image)['image']
    # Channels first
    im = torch.from_numpy(im).float().permute([2, 0, 1])
    return im


def predict_nutrition_score(model, image):
    preprocessed = preprocess_rgb_input(image).unsqueeze(dim=0)
    predictions = model(preprocessed).squeeze()
    label_num = torch.argmax(predictions).item()
    label_probs = torch.softmax(predictions, dim=0).detach().numpy()
    label_name = CLASS_TO_LABEL[label_num]
    return (label_num, label_name, label_probs)


def load_trained_model(checkpoint, checkpoint_dir, device):
    model = NutriNet(pretrained=False)
    load_checkpoint(model, checkpoint=checkpoint,
                          checkpoint_dir=checkpoint_dir, device=device)
    model.eval()
    return model
