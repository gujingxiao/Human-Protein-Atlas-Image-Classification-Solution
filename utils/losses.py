from fastai.vision import *

def focal_loss(input, target, gamma=2.0):
    assert target.size() == input.size(), \
        'Target size ({}) must be the same as input size ({})'.format(target.size(), input.size())

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss

    return loss.sum(dim=1).mean()


def f1_loss(logits, targets):
    epsilon = 1e-6
    beta = 1
    batch_size = logits.size()[0]

    p = F.sigmoid(logits)
    l = targets
    num_pos = torch.sum(p, 1) + epsilon
    num_pos_hat = torch.sum(l, 1) + epsilon
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + epsilon)
    loss = fs.sum() / batch_size
    return 1 - loss


def focal_f1_combined_loss(logits, targets, alpha=0.5):
    return alpha * focal_loss(logits, targets) + (1 - alpha) * f1_loss(logits, targets)