import torch
device = torch.device("cuda:0")
x = torch.randn(10000, 10000, device=device)
for _ in range(1000000000):
    y = torch.matmul(x, x)
print("Done")