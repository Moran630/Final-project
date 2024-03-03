import torch

def accuracy(logits, labels):
    metric_tensor = {}
    probs = logits.softmax(dim=1)
    pred_labels = probs.argmax(dim=1)
    acc_tensor = (pred_labels == labels).float()
    acc = (pred_labels == labels).float().mean()

    pos_idx = (labels == 1)
    if pos_idx.sum() == 0:
        acc_pos = torch.tensor(0.).to(logits.dtype).cuda()
        acc_pos_tensor = torch.tensor([]).cuda()
    else: 
        pos_preds = pred_labels[pos_idx]
        pos_labels = labels[pos_idx]
        acc_pos_tensor = (pos_preds == pos_labels).float()
        acc_pos = (pos_preds == pos_labels).float().mean()

    neg_idx = (labels == 0)
    if neg_idx.sum() == 0:
        acc_neg = torch.tensor(0.).to(logits.dtype).cuda()
        acc_neg_tensor = torch.tensor([]).cuda()
    else:
        neg_preds = pred_labels[neg_idx]
        neg_labels = labels[neg_idx] 
        acc_neg_tensor = (neg_preds == neg_labels).float()
        acc_neg = (neg_preds == neg_labels).float().mean()

    if torch.isnan(acc.detach().cpu()):
        acc = torch.tensor(0.).to(logits.dtype).cuda()
    if torch.isnan(acc_pos.detach().cpu()):
        acc_pos = torch.tensor(0.).to(logits.dtype).cuda()
    if torch.isnan(acc_neg.detach().cpu()):
        acc_neg = torch.tensor(0.).to(logits.dtype).cuda()

    metric_tensor['acc_tensor'] = acc_tensor
    metric_tensor['acc_pos_tensor'] = acc_pos_tensor
    metric_tensor['acc_neg_tensor'] = acc_neg_tensor
    return acc, acc_pos, acc_neg, metric_tensor


def get_correct(logits, labels):
    probs = logits.softmax(dim=1)
    pred_labels = probs.argmax(dim=1)
    corrects = (pred_labels == labels).float().sum()

    pos_idx = (labels == 1)
    pos_num = pos_idx.sum()
    if pos_num == 0:
        corrects_pos = torch.tensor(0.).to(logits.dtype).cuda()
    else: 
        pos_preds = pred_labels[pos_idx]
        pos_labels = labels[pos_idx]
        corrects_pos = (pos_preds == pos_labels).float().sum()

    neg_idx = (labels == 0)
    neg_num = neg_idx.sum()
    if neg_num == 0:
        corrects_neg = torch.tensor(0.).to(logits.dtype).cuda()
    else:
        neg_preds = pred_labels[neg_idx]
        neg_labels = labels[neg_idx] 
        corrects_neg = (neg_preds == neg_labels).float().sum()
    
    return corrects, corrects_pos, pos_num, corrects_neg, neg_num