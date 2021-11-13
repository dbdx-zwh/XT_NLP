### Update:2021.11.14

今天要交作业了，回来看了一遍自己的实验报告（更像是一篇随笔吧）。这篇报告由于没有模板，我主要记录了我做实验遇到的问题以及在那个时间段找到的解决方案（有一些解决方案其实并不准确），为了记录自己的思维成长，我把当时的所思所想都写下来了。

由于本次实验我不是一口气做完的，中间有几个点卡住了导致有一段时间在查资料，所以我在最前面先总结一下我的实验过程：

本次实验我遇到的最大问题就是训练速度提不上来

我开始认为是GPU和CPU性能的差异，但在colab上用gpu实验后，我发现还是很慢

然后我查找lstm的底层实现，认为他一定使用了我们不知道的什么优化方法，发现他的运算是交给C++来实现的

最后还是感觉有问题，就在github搜了一下其他人的实现方式（借鉴借鉴），发现我的问题出在维度转换上。具体地说是 使用自定义的矩阵和输入相乘 与 使用layer层 的区别

至今还有两点疑问没有解决：

1. 自定义矩阵和输入相乘 与 layer层到底有什么区别？前者开始训练的时候损失比较大我能够理解（初始化的问题），但是为什么两者的训练时间也有差异呢？（前者3-4min/epoch，后者5-6s/epoch）
2. 在实现多层的时候，如果层数超过两层，那么这些添加的层的隐藏层（H与C）的权重共享么？我做了实验，发现没什么区别。。。、

若老师或学长对我的疑问能够提出宝贵的建议，请您联系我：QQ：2804272906 赵文昊

### Update:2021.10.28

怎么说呢？在做实验的时候我的心路历程是：有点难 ==> 就这？ ==> 对不起 我错了

### 有点难

LSTM：一个听起来就很牛逼的名字，但是我们还是先捋清楚模型的计算过程：

每个时间点来一个输入Xt，每个单元保存一个隐藏单元Ht以及一个记忆单元Ct

1. 开始根据这些信息就可以计算输入参数、遗忘参数、输出参数、候选记忆单元

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/F3379AC4-6BD8-43BE-A145-AB1A007CAC81.png" alt="F3379AC4-6BD8-43BE-A145-AB1A007CAC81" style="zoom:50%;" />

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/E0040DC5-17D4-4C68-B9CB-AB378110EFD7.png" alt="E0040DC5-17D4-4C68-B9CB-AB378110EFD7" style="zoom:50%;" />

2. 更新记忆单元

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/5C9A0860-B732-47C3-A082-04CF86660F25.png" alt="5C9A0860-B732-47C3-A082-04CF86660F25" style="zoom:50%;" />

3. 更新隐藏状态

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/74742223-D1B6-440C-BFED-929A49EB7142.png" alt="74742223-D1B6-440C-BFED-929A49EB7142" style="zoom:50%;" />

这里给一张我觉得很好的LSTM流程图：

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/559D7DA2-04D7-4C91-94A4-2268ADF9CDB4.png" alt="559D7DA2-04D7-4C91-94A4-2268ADF9CDB4" style="zoom:50%;" />

### 就这？

捋清楚模型思路后我们发现：其实所有参数都是一步一步算出来的。因此代码逻辑就是一条直线（甚至还有很多都是重复的，比如：计算三个参数）

为了不让这部分太空，就记录一个实现中遇到的问题吧：

###### RuntimeError: Trying to backward through the graph a second time, but the buffers have already been

原因：程序在试图执行backward的时候，发现计算图的**缓冲区已经被释放**。

解决办法：在`backward()`函数中添加参数`retain_graph=True`

### 对不起 我错了

在实现一层LSTM的时候很快就写完了代码，好！开始测试！

先说结论：以后能直接调用api就直接用，千万不要犹豫！

对于助教学长提供的数据集，我的电脑用40分钟跑了3个batch，然后结果：

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/4B4AA845-CD0E-49D9-B27C-826CF03AD898.png" alt="4B4AA845-CD0E-49D9-B27C-826CF03AD898" style="zoom:50%;" />

也许训练结果会变好，但显然我等不到那天了。。。

我想了一下原因：这里只有一层，而且代码能跑通。照理来说代码逻辑应该不会错（如果代码逻辑真的错了，请您在方便的时候告诉我咋错的，让我死能瞑目。QQ:2804272906 姓名：赵文昊）

**在知乎上的提问：为什么pytorch lstm比我们自己写的快？**

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/5A0F23C4-86C3-4C60-A209-480CBB2F7CE3.png" alt="5A0F23C4-86C3-4C60-A209-480CBB2F7CE3" style="zoom:50%;" />

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/2AE8E111-3F94-492F-B5F1-93D4E87687C5.png" alt="2AE8E111-3F94-492F-B5F1-93D4E87687C5" style="zoom:50%;" />

看了一下，回答主要还是硬件问题。但我想官方的LSTM有没有用到什么技术可以提高效率？

```
LSTM源码：vscode ctr+v查看
```

这一部分都是检查所传参数是否合理

```python
def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')
```

forward中（初始化参数以及check）：

```python
orig_input = input
# xxx: isinstance check needs to be in conditional for TorchScript to compile
if isinstance(orig_input, PackedSequence):
    input, batch_sizes, sorted_indices, unsorted_indices = input
    max_batch_size = batch_sizes[0]
    max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
            else:
                hx = self.permute_hidden(hx, sorted_indices)
                self.check_forward_args(input, hx, batch_sizes)
```

```
PS：hx即为我们代码中的Ht以及Ct
```

```python
# 开始套娃
if batch_sizes is None:
    result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                      self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                          self.num_layers, self.dropout, self.training, self.bidirectional)
```

最后输出

```python
output = result[0]
hidden = result[1:]
# xxx: isinstance check needs to be in conditional for TorchScript to compile
if isinstance(orig_input, PackedSequence):
    output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
    return output_packed, self.permute_hidden(hidden, unsorted_indices)
else:
    return output, self.permute_hidden(hidden, unsorted_indices)
```

结果我刚想要看vf.lstm怎么实现的，ctr+v点不进去。。。

后来几经周转在pytorch上找到了源码，VF_lstm是用C++实现的！！！

现在在看这个博主的回答，我终于明白他什么意思了。。。

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/2AE8E111-3F94-492F-B5F1-93D4E87687C5.png" alt="2AE8E111-3F94-492F-B5F1-93D4E87687C5" style="zoom:50%;" />

在这里，我找到了一个网址：专门讲vf.lstm实现的，我就简单总结一下，详细请看：

```
ref: https://blog.csdn.net/qq_23981335/article/details/105429676
```

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/B0D31B12-FBF3-42FD-A453-85AE59B6CB37.png" alt="B0D31B12-FBF3-42FD-A453-85AE59B6CB37" style="zoom:50%;" />

首先我们就看见了，它将集合起来的chunked_gate分成了4个gate，这和python中的操作不谋而合。然后就是根据门来更新细胞的状态和隐向量，随后组成一个tuple返回,再细查这个cell_params，它包括了所有的RNN带有cell的结构，LSTM和GRU这两个带有cell的RNN。

故事到这里就结束了，最后找到的实现不同之处就是用了C++以及对所有的门进行集合处理。不想再探究下去了，毁灭吧。



### 多层LSTM

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/F07840F1-83D5-4621-B84B-6BBE22438CF2.png" alt="F07840F1-83D5-4621-B84B-6BBE22438CF2" style="zoom:30%;" />

简单地说就是上一层的隐层状态作为下一层的输入

但是我在实现的时候想到一个问题：不同层之间的隐藏层（Ht,Ct）参数共享么？理论上说不共享，但你永远不知道机器到底学习了什么。因此我打算做一下实验，还是因为训练时间过长的问题，我只处理了部分训练数据，但是从实验结果看差别不大。如果真的共不共享对实验结果影响不大的话，那显然共享可以节省空间。

### 最后插一句

在实验前，我一直以为搭建模型是最难的，但是其实他大部分都是维数错误，比较好debug。但是在自己完成所有attention模块后，我发现model不是boss，而是其他的处理模块（比如数据预处理，make_batch什么的），也许是不熟练，但是感觉那里用了我更多的时间。不过感谢老师只让我们实现模型就可以！

### Update 2021.11.7

和同学讨论了一下为什么我的训练这么慢，发现问题主要出在矩阵乘法上

矩阵乘法我使用的是 A@B 的写法（一个epoch大概15分钟）

对于矩阵乘法优化我有一个误区：就是认为只有GPU才会进行优化，对于CPU而言，@、dot、matmul都是一样的处理效果，但是当我仅仅把代码中的@换为matmul后，发现训练一个epoch仅需要大概1分钟的时间（我设计的batch_size=128）

最后我在colab上用gpu做了一下实验：训练一个epoch需要10秒钟

![A8629154-83CD-4B55-A4FC-C626E0741965](/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/A8629154-83CD-4B55-A4FC-C626E0741965.png)

在使用gpu实验的时候我遇到了一个问题：

```
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

解决方法比较简单：就是在自己设置的矩阵后面加上 .to(device) 即可

```
例如：self.Wxi0 = (torch.randn(emb_size, n_hidden)*0.01).to(device)
```

日后有时间的话，我也想了解一下矩阵乘法是如何优化的（效果这么明显我开始是没想到的）

### Update 2021.11.12

还是感觉自己的实现出了问题，自己查找了一些其他人实现的lstm（自己只是在我已经实现的基础上进行修改，并没有抄袭），发现问题主要出在：如何进行维度转换？

我最早的实现：使用矩阵进行变换

```
self.Wxi = nn.randn(emb_size, n_hidden)
torch.matmul(self.Wxi, Xt)
```

修改后：使用PyTorch线性层进行转换

```
self.Wxi = nn.Linear(emb_size, n_hidden, bias=False)
```

将所有的矩阵乘法替换为pytorch的layer实现后，发现速度变得极快（在cpu下一个epoch也就是六七秒钟）

为什么会这样？或者说用矩阵乘法进行维度变换和layer层有什么区别？

我猜测：可能是前者矩阵初始化的不太合理，导致训练的ppl很大（7000多，正常情况下为600多），但是时间上为什么会那么长呢？可能与layer的底层实现有关？我不是很明白，若老师或者学长能够解答这个疑问，请联系我（QQ：2804272906 赵文昊）

最后展示一下实验结果：

自己实现 层数为1

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/B19107C0-76A1-4AEC-AEA1-C28E87430F2F.png" alt="B19107C0-76A1-4AEC-AEA1-C28E87430F2F" style="zoom:50%;" />

自己实现 层数为3

<img src="/Users/dbdxzwh/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/2804272906/QQ/Temp.db/14709024-6932-4234-A4AB-A757855B8DF1.png" alt="14709024-6932-4234-A4AB-A757855B8DF1" style="zoom:50%;" />

api

<img src="/Users/dbdxzwh/Desktop/NLP_homework/LSTM/558E526E-6B32-49E4-A42C-D0A2B2E9E041.png" alt="558E526E-6B32-49E4-A42C-D0A2B2E9E041" style="zoom:50%;" />

对比可知：

1. 我们在训练速度上已经和api基本持平（甚至更快）
2. 自己实现的一层的LSTM与api在预测准确度上差不多，但是多层的LSTM比api稍差一些，应该是训练的轮数不够

