from bit import BitLinear
import torch
# example
test_data = torch.randn(2, 3)
model = torch.nn.Sequential(
    BitLinear(3, 10),
    torch.nn.ReLU(),
    BitLinear(10, 4)
)

output = model(test_data)
print(output.shape)
# Output: torch.Size([2, 4])
