# -*- coding:utf-8 -*-
# @time :2023.9.12
# @author : wangfy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss_withany(nn.Module):
    def __init__(self, label_names=['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen'], size_average=True):
        super(MultiTaskLoss_withany, self).__init__()
        # ['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        self.label_names = label_names
        self.any_injury_idx = 0
        self.bowel_idx = 1
        self.extravasation_idx = 2
        self.kidney_idx = [3, 4, 5]
        self.liver_idx = [6, 7, 8]
        self.spleen_idx = [9, 10, 11]

        # self.loss_weight = {'any_injury': 6, 'bowel': 2, 'extravasation': 6, 'kidney': 3, 'liver': 3, 'spleen': 3}
        self.loss_weight = {'any_injury': 1, 'bowel': 1, 'extravasation': 1, 'kidney': 1, 'liver': 1, 'spleen': 1}

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def get_logits(self, logits):
        logits_dict = {}
        logits_any_injury = logits[:, self.any_injury_idx]
        logits_dict['any_injury'] = logits_any_injury
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
            for name in label_info:
                label = label_info[name]
                labels_dict[name][b] = label
        return labels_dict

    def get_losses(self, logits_dict, labels_dict):
        
        losses = {}
        for name in self.label_names:
            logits = logits_dict[name]
            labels = labels_dict[name]
            if name in ['any_injury', 'bowel', 'extravasation']:
                loss_v = self.bce_loss(logits, labels)
            else:
                loss_v = self.ce_loss(logits, labels.long())
            losses[name] = loss_v
        return losses
    
    def calc_loss(self, losses, device):
        loss_value = torch.tensor(0.).to(device)
        for name in self.label_names:
            loss_v = losses[name]
            weight = self.loss_weight[name]
            loss_value += weight * loss_v
        return loss_value

    def forward(self, logits, label_infos):
        device = logits.device
        logits_dict = self.get_logits(logits)
        labels_dict = self.get_labels(label_infos, device)
        losses = self.get_losses(logits_dict, labels_dict)
        loss_value = self.calc_loss(losses, device)
       
        return losses, loss_value


class MultiTaskLoss_withoutany(nn.Module):
    def __init__(self, label_names=['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen'], size_average=True):
        '''
        function get_logits: Get logits from the output of the model
        function get_labels: Get label with ground truth:
                                                        bowel healthy/injury
                                                        extravasation healthy/injury
                                                        kidney healthy/low injury/high injury
                                                        liver healthy/low injury/high injury
                                                        spleen healthy/low injury/high injury
        function calc_loss: According to the given weights(class_weight_binary, class_weight_triple), model prediction results and labels, calculate the binary-class and three-class loss functions respectively.
        '''
        super(MultiTaskLoss_withoutany, self).__init__()
        # ['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        self.label_names = label_names
        self.bowel_idx = 0
        self.extravasation_idx = 1
        self.kidney_idx = [2, 3, 4]
        self.liver_idx = [5, 6, 7]
        self.spleen_idx = [8, 9, 10]

        # self.loss_weight = {'any_injury': 6, 'bowel': 2, 'extravasation': 6, 'kidney': 3, 'liver': 3, 'spleen': 3}
        self.loss_weight = {'any_injury': 1, 'bowel': 1, 'extravasation': 1, 'kidney': 1, 'liver': 1, 'spleen': 1}

        self.class_weight_binary = {'bowel': {0: 1, 1: 2}, 'extravasation': {0: 1, 1: 6}}
        self.class_weight_triple = {'kidney': [1, 2, 4], 'liver': [1, 2, 4], 'spleen': [1, 2, 4]}

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_prob_loss = nn.BCELoss()
        # self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_dict = {}
        for name in self.class_weight_triple:
            weight = self.class_weight_triple[name]
            weight = torch.tensor(weight).float().cuda()
            self.ce_loss_dict[name] = nn.CrossEntropyLoss(weight=weight,reduction="sum")


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
            for name in label_info:
                label = label_info[name]
                labels_dict[name][b] = label
        return labels_dict

    def get_binary_weight(self, labels, name):
        weight = torch.zeros_like(labels).to(labels)
        class_weight = self.class_weight_binary[name]
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
            if name in ['bowel', 'extravasation']:
                probs =  logits.sigmoid().unsqueeze(dim=1)
            else:
                probs_softmax = logits.softmax(dim=1)
                probs = 1.0 - probs_softmax[:, 0]
                probs = probs.unsqueeze(dim=1)
            probs_list.append(probs)
        probs_list = torch.cat(probs_list, dim=1)
        probs_injury, ind = probs_list.max(dim=1, keepdim=False)
        loss = self.bce_prob_loss(probs_injury, labels.float())
        return loss

    def get_losses(self, logits_dict, labels_dict, device):
        losses = {}
        for name in self.label_names:
            if name not in logits_dict:
                assert name == 'any_injury'
                # losses[name] = torch.torch.tensor(0.).to(device)
                losses[name] = self.get_any_injury_loss(logits_dict, labels_dict[name])
                continue
            logits = logits_dict[name]
            labels = labels_dict[name]
            if name in ['bowel', 'extravasation']:
                weight = self.get_binary_weight(labels, name)
                loss_v_batch = self.bce_loss(logits, labels)
                loss_v = weight * loss_v_batch
                # print(loss_v_batch, weight, loss_v)
                loss_v = loss_v.mean()
                # print(loss_v)
            else:
                ce_loss = self.ce_loss_dict[name]
                loss_v = ce_loss(logits, labels.long())
            losses[name] = loss_v
        return losses
    

    def calc_loss(self, losses, device):
        loss_value = torch.tensor(0.).to(device)
        for name in self.label_names:
            loss_v = losses[name]
            weight = self.loss_weight[name]
            loss_value += weight * loss_v
        return loss_value

    def forward(self, logits, label_infos):
        device = logits.device
        logits_dict = self.get_logits(logits)
        labels_dict = self.get_labels(label_infos, device)
        losses = self.get_losses(logits_dict, labels_dict, device)
        loss_value = self.calc_loss(losses, device)
       
        return losses, loss_value


class SingleTaskLoss(nn.Module):
    def __init__(self, label_names=['spleen'], size_average=True):
        super(SingleTaskLoss, self).__init__()
        # ['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        self.label_names = label_names
        assert len(self.label_names) == 1

        # weight = None
        # weight = torch.tensor([1.0,2.0,4.0]).cuda()
        weight = torch.tensor([1.0,6.0]).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight,reduction="sum")


    def get_logits(self, logits):
        logits_dict = {}
        logits_dict[self.label_names[0]] = logits

        return logits_dict

    def get_labels(self, label_infos, device):
        labels_dict = {}
        bs = len(label_infos)
        for name in self.label_names:
            labels_dict[name] = torch.zeros((bs, )).float().to(device)
        for b in range(bs):
            label_info = label_infos[b]
            for name in label_info:
                if name not in self.label_names:
                    continue
                label = label_info[name]
                labels_dict[name][b] = label
        return labels_dict


    def get_losses(self, logits_dict, labels_dict, device):
        losses = {}
        for name in self.label_names:
            logits = logits_dict[name]
            labels = labels_dict[name]
            loss_v = self.ce_loss(logits, labels.long()) / logits.size()[0]
            losses[name] = loss_v
        return losses
    
    def calc_loss(self, losses, device):
        # loss_value = losses[self.label_names[0]]
        # for name in self.label_names[1:]:
        #     loss_v = losses[name]
        #     loss_value += loss_v
        # return loss_value

        loss_value = torch.tensor(0.).to(device)
        for name in self.label_names:
            loss_v = losses[name]
            loss_value += loss_v
        return loss_value

    def forward(self, logits, label_infos):
        device = logits.device
        logits_dict = self.get_logits(logits)
        labels_dict = self.get_labels(label_infos, device)
        losses = self.get_losses(logits_dict, labels_dict, device)
        loss_value = self.calc_loss(losses, device)
       
        return losses, loss_value


class SingleTaskLossSigmoid(nn.Module):
    def __init__(self, label_names=['spleen'], num_classes=3, size_average=True):
        super(SingleTaskLossSigmoid, self).__init__()
        # ['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        self.label_names = label_names
        assert len(self.label_names) == 1

        # weight = None
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes

    def get_logits(self, logits):
        logits_dict = {}
        logits_dict[self.label_names[0]] = logits

        return logits_dict

    def get_labels(self, label_infos, device):
        labels_dict = {}
        bs = len(label_infos)
        for name in self.label_names:
            labels_dict[name] = torch.zeros((bs, self.num_classes)).float().to(device)
        for b in range(bs):
            label_info = label_infos[b]
            for name in label_info:
                if name not in self.label_names:
                    continue
                label = label_info[name]
                labels_dict[name][b][label] = 1
        return labels_dict


    def get_losses(self, logits_dict, labels_dict, device):
        losses = {}
        for name in self.label_names:
            logits = logits_dict[name]
            labels = labels_dict[name]
           
            loss_v = self.bce_loss(logits, labels)
            losses[name] = loss_v
        return losses
    
    def calc_loss(self, losses, device):
        loss_value = torch.tensor(0.).to(device)
        for name in self.label_names:
            loss_v = losses[name]
            loss_value += loss_v
        return loss_value

    def forward(self, logits, label_infos):
        device = logits.device
        logits_dict = self.get_logits(logits)
        labels_dict = self.get_labels(label_infos, device)
        losses = self.get_losses(logits_dict, labels_dict, device)
        loss_value = self.calc_loss(losses, device)
       
        return losses, loss_value

if __name__ == '__main__':
    torch.random.manual_seed(42)
    multi_task_loss = SingleTaskLossSigmoid()
    probs = torch.rand((2, 3))
    print(probs)
    logits = probs
    logits[:, :3] = torch.log(logits[:, :3] / (1- logits[:, :3]))
    print(logits)
    label_infos = [{'any_injury': 1, 'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 1, 'spleen': 1}, 
                   {'any_injury': 0, 'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}]
    
    losses, loss_value = multi_task_loss(logits, label_infos)
    print(losses)
    print(loss_value)
