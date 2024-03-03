import torch

# bowel_healthy,bowel_injury,extravasation_healthy,extravasation_injury,kidney_healthy,kidney_low,kidney_high,liver_healthy,liver_low,liver_high,spleen_healthy,spleen_low,spleen_high

def get_model_result(logits, data_region, single_loss='softmax_ce'):
    ''''
    model_outpus: shape: (batch_size, num_classes)
    0: bowel (sigmoid: healthy/injury)
    1: extravasation (sigmoid: healthy/injury)
    2-4: kidney (softmax: healthy/low/high)
    5-7: liver (softmax: healthy/low/high)
    8-10: spleen (softmax: healthy/low/high)
    '''
    bs, num_classes = logits.size()
    if data_region == 'all':
        logits_bowel = logits[:, 0]
        logits_extravasation = logits[:, 1]
        logits_kidney = logits[:, 2: 5]
        logits_liver = logits[:, 5: 8]
        logits_spleen = logits[:, 8: 11]

        probs_bowel = logits_bowel.sigmoid().unsqueeze(1)
        probs_extravasation = logits_extravasation.sigmoid().unsqueeze(1)
        probs_kidney = logits_kidney.softmax(dim=1)
        probs_liver = logits_liver.softmax(dim=1)
        probs_spleen = logits_spleen.softmax(dim=1)

        probs_bowel_healthy = 1.0 - probs_bowel
        probs_extravasation_healthy = 1.0 - probs_extravasation
        # print()
        output_probs = torch.cat([probs_bowel_healthy, probs_bowel, probs_extravasation_healthy, probs_extravasation, probs_kidney, probs_liver, probs_spleen], dim=1)
    else:
        # single cls, data_region: spleen | liver | kidney
        if single_loss=='softmax_ce':
            output_probs = logits.softmax(dim=1)
        else:
            outputs = logits.sigmoid()
            probs_sum = outputs.sum(dim=1)
            output_probs = outputs / probs_sum.unsqueeze(1)
    return output_probs

    
