from tools import TwoHotDistSymlog
import torch

torch.set_printoptions(linewidth=180)

logits = [-0.0] * 10
logits = torch.tensor(logits).to("cuda")
logits.requires_grad = True
opt = torch.optim.AdamW([logits], lr=3e-5)
dist = TwoHotDistSymlog(logits, device="cuda", buckets=10, low=-3, high=3)

for i in range(10000):
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
    if i % 200 == 0:
        print(logits)