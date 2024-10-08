import torch
import torch.nn as nn
from torch.autograd import grad

# 定义神经网络模型
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedNN, self).__init__()
        # 3层隐藏层，大小为20
        self.hidden1 = nn.Linear(2, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.hidden3 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)  # 将x和t合并作为输入
        out = self.activation(self.hidden1(input))
        out = self.activation(self.hidden2(out))
        out = self.activation(self.hidden3(out))
        u = self.output_layer(out)
        return u

# 定义守恒方程的物理约束
def conservation_law_loss(model, x, t):
    u = model(x, t)
    u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
    f = u_t + 0.25 * u * u_x  # 方程的物理约束部分
    return f

# 初始条件
def initial_condition(x):
    u0 = torch.zeros_like(x)
    u0[(x > 0) & (x < 1)] = 1
    return u0

# 边界条件
def boundary_condition(x, t):
    return torch.zeros_like(x)

# 生成训练数据
def generate_data(Nx, Nt):
    x = torch.linspace(-1, 6, Nx)
    t = torch.linspace(0, 10, Nt)
    x, t = torch.meshgrid(x, t, indexing='ij')
    return x.flatten().reshape(-1, 1), t.flatten().reshape(-1, 1)

# L2相对误差计算
def L2RE(pred, true):
    return torch.sqrt(torch.mean((pred - true)**2)) / torch.sqrt(torch.mean(true**2))

# 检查GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置训练数据和模型
Nx, Nt = 100, 100
x_train, t_train = generate_data(Nx, Nt)

# 将数据转换为GPU上的张量
x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)

# 初始化模型并将其移动到GPU
model = PhysicsInformedNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 物理损失
    f = conservation_law_loss(model, x_train, t_train)
    loss_phys = torch.mean(f**2)

    # 初始条件损失
    x_ic = torch.linspace(-1, 6, Nx).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = initial_condition(x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)

    # 边界条件损失
    u_bc_pred = model(x_ic, torch.full_like(x_ic, 0).to(device))
    loss_bc = torch.mean((u_bc_pred - boundary_condition(x_ic, t_ic))**2)

    # 总损失
    loss = loss_phys + loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 预测结果
x_test = torch.linspace(-1, 6, Nx).to(device)
t_test = torch.linspace(0, 10, Nt).to(device)
x_test, t_test = torch.meshgrid(x_test, t_test, indexing='ij')
u_pred = model(x_test.flatten().reshape(-1, 1), t_test.flatten().reshape(-1, 1))

# L2RE计算
# u_true = initial_condition(x_test.flatten().reshape(-1, 1)).to(device)
# L2_error = L2RE(u_pred, u_true)
# print(f"L2 Relative Error: {L2_error.item()}")