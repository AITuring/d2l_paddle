
## PyTorch安装

Mac采用pip3命令：

```bash
# Python 3.x
pip3 install torch torchvision
```

如果中间`pip`报下面的错：

```bash
Traceback (most recent call last):
  File "/usr/local/bin/pip3", line 5, in <module>
    from pip._internal.cli.main import main
ModuleNotFoundError: No module named 'pip._internal.cli.main'
```

就重装`pip`: 

```bash
python3 -m pip install --upgrade pip
```
安装完成之后，可以用以下命令验证是否安装成功：

```python
import torch
x = torch.rand(5, 3)
print(x)
```

如果没有报错，且输出类似于下面，说明安装成功了！：

```python
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])

```

## PaddlePaddle安装

进入[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/macos-pip.html),选择适合自己的命令安装即可

安装之后可以使用`python`进入`python`解释器，输入`import paddle` ，再输入 `paddle.utils.run_check()`。如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。