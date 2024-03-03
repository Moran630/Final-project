import torch

# bowel_healthy,bowel_injury,extravasation_healthy,extravasation_injury,kidney_healthy,kidney_low,kidney_high,liver_healthy,liver_low,liver_high,spleen_healthy,spleen_low,spleen_high

def get_model_result(logits):
    logits_bowel = logits[:, :2]
    logits_extravasation = logits[:, 2:4]
    logits_kidney = logits[:, 4:7]
    logits_liver = logits[:, 7:10]
    logits_spleen = logits[:, 10:13]
    probs_bowel = logits_bowel.softmax(dim=1)
    probs_extravasation = logits_extravasation.softmax(dim=1)
    probs_kidney = logits_kidney.softmax(dim=1)
    probs_liver = logits_liver.softmax(dim=1)
    probs_spleen = logits_spleen.softmax(dim=1)

    output_probs = torch.cat([probs_bowel, probs_extravasation, probs_kidney, probs_liver, probs_spleen], dim=1)
    return output_probs

    
