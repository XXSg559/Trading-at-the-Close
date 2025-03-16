import torch
from bit.bitlinear import BitLinear
# 解决XOR问题
# 1. 定义数据
data_x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
data_y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
# 2. 定义模型
model = torch.nn.Sequential(
    BitLinear(2, 10),
    torch.nn.ReLU(),
    BitLinear(10, 1)
)
# 3. 定义损失函数
criterion = torch.nn.MSELoss()
# 4. 定义优化器
optimizer = torch.optim.Adam(model.parameters())
# 5. 训练模型
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(data_x)
    loss = criterion(output, data_y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'epoch: {epoch}, loss: {loss.item()}')
# 6. 测试模型
output = model(data_x)