import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.dpt import DPT
from utils.logger import setup_logger
from utils.visualization import visualize_segmentation

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--output', type=str, help="Path to save output image")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(config['log_dir'])

    # Transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config['mean'], config['std'])])

    # Load image
    image = load_image(args.image, transform).to(device)

    # Model
    model = DPT(num_classes=config['num_classes'], backbone=config['backbone'], pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(image)
        output = torch.argmax(output, dim=1)

    # Visualization
    visualize_segmentation(image.cpu(), output.cpu(), config['label_colors'], args.output)

if __name__ == "__main__":
    main()