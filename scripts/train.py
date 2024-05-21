import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.dpt import DPT
from data.voc import VOCDataset
from utils.logger import setup_logger
from utils.metrics import AverageMeter
from utils.metrics import accuracy


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, logger):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        # 移除目标张量的单一通道维度
        targets = targets.squeeze(1)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, targets)
        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

        if i % 10 == 0:
            logger.info(
                f'Epoch [{epoch}], Step [{i}/{len(train_loader)}], Loss: {losses.avg:.4f}, Accuracy: {accs.avg:.4f}')


def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(config['log_dir'])

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all labels to 256x256
        transforms.ToTensor()
    ])

    if config['dataset'] == 'voc':
        train_dataset = VOCDataset(config['data_dir'], split='train', transform=transform,
                                   target_transform=target_transform)
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'])

    # Model
    model = DPT(num_classes=config['num_classes'], backbone=config['backbone'], pretrained=config['pretrained'])
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    # Training Loop
    for epoch in range(config['epochs']):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, logger)

        # Save the model checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = f"{config['checkpoint_dir']}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()