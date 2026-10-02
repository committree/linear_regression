# linear_regression
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(100,1)
y = 3 * x + 2 + 0.1 *torch.randn(100,1)

lr = 0.01
epochs = 200

# 原始版本
w = torch.randn(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

print("===== 原始手写梯度版训练开始 =====")
for epoch in range(epochs):
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).sum()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.gard.zero_()

    if (epoch + 1) % 20 == 0:
        print(epoch + 1, loss.item())

print(f"\n训练完成: w = {w.item():.4f},b = {b.item():.4f}")

# nn.Module框架版本
model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

print("\n===== 框架版训练开始 =====")
for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(epoch + 1, loss.item())

trained_w = model.weight.item()
trained_b = model.bias.item()
print(f"\n训练完成: w = {trained_w:.4f}, b = {trained_b:.4f}")
