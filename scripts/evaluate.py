import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.dpt import DPT
from data.voc import VOCDataset
from utils.logger import setup_logger
from utils.metrics import AverageMeter
from utils.metrics import accuracy
from utils.metrics import compute_iou


def evaluate(model, data_loader, criterion, device, logger):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    ious = AverageMeter()

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc = accuracy(outputs, targets)
            iou = compute_iou(outputs, targets)

            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))
            ious.update(iou, images.size(0))

    logger.info(f"Validation Loss: {losses.avg:.4f}, Accuracy: {accs.avg:.4f}, IoU: {ious.avg:.4f}")
def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(config['log_dir'])

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config['mean'], config['std'])])
    if config['dataset'] == 'voc':
        val_dataset = VOCDataset(config['data_dir'], split='val', transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Model
    model = DPT(num_classes=config['num_classes'], backbone=config['backbone'], pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    evaluate(model, val_loader, criterion, device, logger)

if __name__ == "__main__":
    main()