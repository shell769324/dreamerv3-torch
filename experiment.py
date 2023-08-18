from tools import TwoHotDistSymlog
import torch

logits = [-0.0] * 10
logits = torch.tensor(logits)
logits.requires_grad = True
opt = torch.optim.AdamW([logits])
dist = TwoHotDistSymlog(logits, device="cuda", buckets=10, low=-3, high=3)

for i in range(200):
    if i % 8 == 0:
        ground = torch.tensor([-0.5]).to("cuda")
    elif i % 23 == 1:
        ground = torch.tensor([1]).to("cuda")
    elif i % 8 == 2:
        ground = torch.tensor([0.5]).to("cuda")
    else:
        ground = torch.tensor([0]).to("cuda")
    loss = -dist.log_prob(ground).mean()
    loss.backward()
    opt.step()
    print(logits)