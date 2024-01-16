import argparse

import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image

from defaultargs import defaultargs

from reshape import reshape_model

parser = argparse.ArgumentParser(description='Run Image Classifier')

parser.add_argument('--img', type=str, default='data/test/AppleScab1.JPG', 
                    help='path to desired image to run model on'
					'image path (default: data/test/AppleScab1.JPG/)')

args = parser.parse_args()

def create_image_tensor(img_path):
    img = Image.open(img_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    val_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])

    t = val_transforms(img)

    t = t.unsqueeze(0)

    return t

def main(args):
    img_tensor = create_image_tensor(args.img)

    model = torchvision.models.resnet18()
    model = reshape_model(model, 33)
    model.load_state_dict(torch.load("models/model_best.pth.tar")['state_dict'])
    model.eval()

    output = model(img_tensor)
    prediction = torch.argmax(output)
    confidence = torch.max(torch.nn.Softmax(dim=1)(output)) * 100
    print(prediction.numpy(), confidence.detach().numpy())

if __name__ == "__main__":
    for key in defaultargs.keys():
        args.__dict__[key] = defaultargs[key]

    main(args)