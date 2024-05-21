import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).float().sum()
        total = torch.tensor(target.size(0) * target.size(1) * target.size(2)).float()
        acc = correct / total * 100.0
    return acc.item()

def compute_iou(output, target, num_classes=21):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        ious = []
        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        iou = torch.tensor(ious).nanmean().item() * 100.0
    return iou

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count