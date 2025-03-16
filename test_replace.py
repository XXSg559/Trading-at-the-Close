import torch
import torch.nn as nn
from bit.bitlinear import BitLinear
# 假设你已经定义了 BitLinear


# 替换函数
def replace_linear(module):
    if isinstance(module, nn.Linear):
        # 替换为 BitLinear
        bitlinear = BitLinear(module.in_features, module.out_features, module.bias is not None)
        # 复制权重和偏置
        bitlinear.weight = module.weight
        if module.bias is not None:
            bitlinear.bias = module.bias
        return bitlinear
    else:
        return module

# 示例模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 40)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 创建模型
model = MyModel()

# 替换所有 nn.Linear 为 BitLinear
model = model.apply(replace_linear)

# 测试
x = torch.randn(2, 10)
output = model(x)
print(output.shape)  # 输出: torch.Size([2, 40])