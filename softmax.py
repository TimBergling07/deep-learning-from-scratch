import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
lr = 0.1

W = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad = True)
b = torch.zeros(num_outputs, requires_grad = True)

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim = True), X.sum(1, keepdim = True) #求和

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0, 0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
       self.data = [0, 0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    def __init__(self, xlabel = None, ylabel=None, legend=None, xlim=None, 
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows = 1, ncols = 1,
                 figsize = (3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrow, ncols, figsize = figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(self.aexs[0],
            xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)

def softmax(X)
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp / partition #这里用到了广播机制

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)

def net(x):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6],[0.3, 0.2, 0.5]])
y_hat[[0, 1], y] #取第一行第一列和第二行第三列

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

loss = cross_entropy

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def accuracy(y_hat, y):
    """计算预测正确的数量""""
    #len返回y_hat.shape这个元组的长度，与1比较大小就是判断是不是二维矩阵
    #[1]返回的是num_classes,即分类数，确保至少是二分类
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
 
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metirc = Accumulator(2) #正确预测数，预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

evaluate_accuracy(net, test_iter)

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y), y.size().numel()
            ) #总损失，正确预测数，样本总数
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y))
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metircs + (test.acc,))
    train_loss, train_acc = train_metrics

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for(X, y in test_iter):
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, and preds)]
    d2l.show_images(X[0:n].reshape(n, 28, 28), 1, n, titles = titles[0:n])

predict_ch3(net, test_iter)