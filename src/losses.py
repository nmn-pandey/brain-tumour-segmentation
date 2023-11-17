import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
# FOR 4-dimensional tensors i.e. the ones without one-hot-encoding

def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true[:, 1] * y_pred[:, 1])
    union = torch.sum(y_true[:, 1]) + torch.sum(y_pred[:, 1])
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true[:, 2] * y_pred[:, 2])
    union = torch.sum(y_true[:, 2]) + torch.sum(y_pred[:, 2])
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true[:, 3] * y_pred[:, 3])
    union = torch.sum(y_true[:, 3]) + torch.sum(y_pred[:, 3])
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

# Dice Loss
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = torch.sum(y_true * y_pred, axis=[1, 2, 3])
    union = torch.sum(y_true, [1, 2, 3]) + torch.sum(y_pred, axis=[1, 2, 3])
    dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
    return dice

# Define Focal loss
def categorical_focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    epsilon = 1e-5
    y_pred = torch.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * torch.log(y_pred)
    focal = alpha * torch.pow(1 - y_pred, gamma) * cross_entropy
    return torch.mean(focal, dim=0)

# Define total loss
def total_loss(y_true, y_pred, wt1 = 1, wt2 = 2):
    dice = dice_coef(y_true, y_pred)
    focal = categorical_focal_loss(y_true, y_pred)
    total = wt1 * dice + wt2 * focal
    return total


# def dice_coef(y_true, y_pred, epsilon=0.00001):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
#     """
#     axis = (1, 2, 3)  # For 3D data (depth, height, width)
#     # axis = (1, 2)   # For 2D data (height, width)

#     intersection = 2. * torch.sum(y_true * y_pred, dim=axis) + epsilon
#     denominator = torch.sum(y_true * y_true, dim=axis) + torch.sum(y_pred * y_pred, dim=axis) + epsilon
#     return torch.mean((intersection) / (denominator))

def accuracy(outputs, masks):
    correct = (outputs == masks).float().sum()
    total = outputs.numel()
    return correct / total

def sensitivity(outputs, masks, class_id, epsilon=1e-7):
    true_positives = ((outputs == class_id) & (masks == class_id)).float().sum()
    false_negatives = ((outputs != class_id) & (masks == class_id)).float().sum()
    return true_positives / (true_positives + false_negatives + epsilon)

def specificity(outputs, masks, class_id, epsilon=1e-7):
    true_negatives = ((outputs != class_id) & (masks != class_id)).float().sum()
    false_positives = ((outputs == class_id) & (masks != class_id)).float().sum()
    return true_negatives / (true_negatives + false_positives + epsilon)

def precision(outputs, masks, class_id, epsilon=1e-7):
    true_positives = ((outputs == class_id) & (masks == class_id)).float().sum()
    false_positives = ((outputs == class_id) & (masks != class_id)).float().sum()
    return true_positives / (true_positives + false_positives + epsilon)

#def dice_score(outputs, masks, class_id, epsilon=1e-7):
#    true_positives = ((outputs == class_id) & (masks == class_id)).float().sum()
#    false_positives = ((outputs == class_id) & (masks != class_id)).float().sum()
#    false_negatives = ((outputs != class_id) & (masks == class_id)).float().sum()
#    return 2 * true_positives / (2 * true_positives + false_positives + false_negatives + epsilon)

def dice_score(outputs, masks, class_id, epsilon=1e-7):
    y_true = (masks == class_id)
    y_pred = (outputs == class_id)
    intersection = (y_true & y_pred).float().sum()
    union = y_true.float().sum() + y_pred.float().sum()
    return (2. * intersection + epsilon) / (union + epsilon)

class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, num_classes=4, class_weights=None, device_num=0):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.num_classes = num_classes

        if class_weights is not None:
            self.num_classes = len(class_weights)
            device = torch.device(f"cuda:{device_num}")  # For GPU
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Ensure weights are float and on the correct device

        self.ce_loss = CrossEntropyLoss(weight=class_weights)
        print(f"num_classes : {self.num_classes}, class_weights : {class_weights}")

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = F.softmax(pred, dim=1)
        total_loss = 0
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls, ...]
            true_cls = (target == cls).float()

            intersection = torch.sum(pred_cls * true_cls)
            cardinality = torch.sum(pred_cls + true_cls)
            dice = (2. * intersection + smooth) / (cardinality + smooth)
            total_loss += (1 - dice)

        return total_loss / self.num_classes

    def forward(self, logits, targets):
        # Cross Entropy Loss
        #ce_loss = self.ce_loss(logits, targets)
        ce_loss = self.ce_loss(torch.clamp(logits, min=-100, max=100), targets)

        # Dice Loss
        dice = self.dice_loss(logits, targets)

        return self.weight_ce * ce_loss + self.weight_dice * dice
