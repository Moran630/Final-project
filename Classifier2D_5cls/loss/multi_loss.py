# -*- coding:utf-8 -*-
# @time :2023.09.12
# @author :wangfy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, label_names=['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury'], use_gpu=True, label_smoothing=0.0):
        super(MultiTaskLoss, self).__init__()
        # ['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        self.label_names = label_names
        self.bowel_idx = [0, 1]
        self.extravasation_idx = [2, 3]
        self.kidney_idx = [4, 5, 6]
        self.liver_idx = [7, 8, 9]
        self.spleen_idx = [10, 11, 12]

        if use_gpu:
            weight_bowel = torch.tensor([1., 2.]).float().cuda()
            weight_extravasation = torch.tensor([1., 6.]).float().cuda()
            weight_kdiney = torch.tensor([1., 2., 4.]).float().cuda()
            weight_liver = torch.tensor([1., 2., 4.]).float().cuda()
            weight_spleen = torch.tensor([1., 2., 4.]).float().cuda()
        else:
            weight_bowel = torch.tensor([1., 2.]).float()
            weight_extravasation = torch.tensor([1., 6.]).float()
            weight_kdiney = torch.tensor([1., 2., 4.]).float()
            weight_liver = torch.tensor([1., 2., 4.]).float()
            weight_spleen = torch.tensor([1., 2., 4.]).float()

        
        self.ce_loss = {
            'bowel': nn.CrossEntropyLoss(weight_bowel, reduction='none', label_smoothing=label_smoothing), 
            'extravasation': nn.CrossEntropyLoss(weight_extravasation, reduction='none', label_smoothing=label_smoothing),
            'kidney': nn.CrossEntropyLoss(weight_kdiney, reduction='none', label_smoothing=label_smoothing),
            'liver': nn.CrossEntropyLoss(weight_liver, reduction='none', label_smoothing=label_smoothing),
            'spleen': nn.CrossEntropyLoss(weight_spleen, reduction='none', label_smoothing=label_smoothing)
        }

        self.bce_prob_loss = nn.BCELoss(reduction='none')
        self.any_injury_weight = {0: 1, 1: 6}

    def get_logits(self, logits):
        logits_dict = {}

        logits_bowel = logits[:, self.bowel_idx]
        logits_dict['bowel'] = logits_bowel
        logits_extravasation = logits[:, self.extravasation_idx]
        logits_dict['extravasation'] = logits_extravasation
        logits_kidney = logits[:, self.kidney_idx]
        logits_dict['kidney'] = logits_kidney
        logits_liver = logits[:, self.liver_idx]
        logits_dict['liver'] = logits_liver
        logits_spleen = logits[:, self.spleen_idx]
        logits_dict['spleen'] = logits_spleen

        return logits_dict

    def get_labels(self, label_infos, device):
        labels_dict = {}
        bs = len(label_infos)
        for name in self.label_names:
            labels_dict[name] = torch.zeros((bs, )).float().to(device)
        for b in range(bs):
            label_info = label_infos[b]
            any_injury = 0
            for name in self.label_names:
                if name == 'any_injury':
                    continue
                label = label_info[name]
                if label != 0:
                    any_injury = 1
                labels_dict[name][b] = label
            labels_dict['any_injury'][b] = any_injury
        return labels_dict

    def get_any_injury_weight(self, labels):
        weight = torch.zeros_like(labels).to(labels)
        class_weight = self.any_injury_weight
        for i, label in enumerate(labels):
            w = class_weight[label.cpu().numpy().item()]
            weight[i] = w
        return weight

    def get_any_injury_loss(self, logits_dict, labels):
        probs_list = []
        for name in self.label_names:
            if name not in logits_dict:
                continue
            logits = logits_dict[name]
            probs_softmax = logits.softmax(dim=1)
            probs = 1.0 - probs_softmax[:, 0]
            probs = probs.unsqueeze(dim=1)
            probs_list.append(probs)
        probs_list = torch.cat(probs_list, dim=1)
        probs_injury, ind = probs_list.max(dim=1, keepdim=False)
        weight = self.get_any_injury_weight(labels)
        loss = self.bce_prob_loss(probs_injury, labels.float()) * weight
        return loss.mean()


    def get_losses(self, logits_dict, labels_dict, bs):
        losses = {}
        for name in self.label_names:
            if name not in logits_dict:
                assert name == 'any_injury'
                # losses[name] = torch.torch.tensor(0.).to(device)
                losses[name] = self.get_any_injury_loss(logits_dict, labels_dict[name])
                continue
            logits = logits_dict[name]
            labels = labels_dict[name]
            loss_v = self.ce_loss[name](logits, labels.long())
            losses[name] = loss_v.sum() / bs
        return losses
    
    def calc_loss(self, losses, device):
        loss_value = torch.tensor(0.).to(device)
        for name in self.label_names:
            loss_v = losses[name]
            loss_value += loss_v
        return loss_value

    def forward(self, logits, label_infos):
        bs = logits.size(0)
        device = logits.device
        logits_dict = self.get_logits(logits)
        labels_dict = self.get_labels(label_infos, device)
        losses = self.get_losses(logits_dict, labels_dict, bs)
        loss_value = self.calc_loss(losses, device)
       
        return losses, loss_value


if __name__ == '__main__':
    torch.random.manual_seed(42)
    multi_task_loss = MultiTaskLoss(use_gpu=False)
    logits = torch.rand((4, 13))
    print(logits)
    label_infos = [{'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}, 
                   {'bowel': 1, 'extravasation': 0, 'kidney': 1, 'liver': 2, 'spleen': 0}, 
                   {'bowel': 0, 'extravasation': 1, 'kidney': 2, 'liver': 0, 'spleen': 1},
                   {'bowel': 1, 'extravasation': 1, 'kidney': 0, 'liver': 1, 'spleen': 2}]
    
    losses, loss_value = multi_task_loss(logits, label_infos)
    print(losses)
    print(loss_value)