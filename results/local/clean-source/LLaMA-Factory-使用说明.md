# 安装结果

- WSL 2.6.3，发行版使用 WSL2：原已安装。
- 发行版名称 `Ubuntu`，实际系统 Ubuntu 22.04.5 LTS：原已安装。
- LLaMA-Factory 0.9.6.dev0：已安装到 `/home/lgh/LLaMA-Factory`。
- Python 3.11.17 虚拟环境：`/home/lgh/.venvs/llamafactory`。
- PyTorch 2.14.1+cu130，bitsandbytes 0.50.2。
- 依赖兼容性检查通过；RTX 4070 Laptop GPU 的 CUDA 矩阵运算通过；WebUI HTTP 200。

# 从 Windows 启动

在 PowerShell 运行：

```powershell
& 'C:\Users\lgh\Documents\ChatGPT\NLP\start-llamafactory.ps1'
```

随后访问 http://127.0.0.1:7860 。保持终端运行，按 Ctrl+C 停止。

# 手动进入 Ubuntu

```powershell
wsl -d Ubuntu -u lgh
```

在 Linux 终端运行：

```bash
cd /home/lgh/LLaMA-Factory
source /home/lgh/.venvs/llamafactory/bin/activate
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```

模型权重和训练数据尚未下载，尚未进行实际模型训练。安装使用官方 GitHub 源码，依赖下载使用清华 PyPI 镜像，未修改全局软件源。
