# 一、服务器配置

## 1.1 使用守护进程终端

参考资料：
> https://www.autodl.com/docs/daemon/

在终端下，执行以下命令：

```sh
# 安装 screen
apt-get update && apt-get install -y screen

# 解决 screen 终端中文乱码问题
echo "defencoding UTF-8" >>  ~/.screenrc
echo "encoding UTF-8" >>  ~/.screenrc

# 进入 screen 终端，创建一个新的 screen 会话
screen -U
```

Q1: 如何中途离开 screen 会话（不退出）？

A: 使用快捷键 `Control + A + D`。

Q2: 如何重新进入 screen 会话？

A: 在终端下，执行以下命令：

```sh
screen -ls
screen -U -r 2564.pts-0.autodl-container-7b6649a2d0-b8382948
```

Q3：如何完全退出 screen 会话？

A: 如果有正在执行的任务，使用快捷键 `Control + C` 结束任务。

然后使用快捷键 `Control + D` 完全退出 screen 会话。

## 1.1 清理系统盘空间

参考资料：
> https://www.autodl.com/docs/qa1/

## 1.2 设置网络代理

参考资料：
> https://www.autodl.com/docs/network_turbo/

在终端下，执行以下命令，设置 AutoDL 网络代理：

```sh
source /etc/network_turbo
```

在终端下，执行以下命令，取消 AutoDL 网络代理：

```sh
unset http_proxy && unset https_proxy
```

## 1.3 安装 uv 与项目依赖包

在终端下，执行以下命令：

```sh
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使 uv 命令立即生效
source ~/.local/bin/env

# 查看 uv 版本
uv --version
```

Q: 如何修改 uv 缓存文件目录？

A: 在 `~/.bashrc` 文件的末尾，追加以下内容并保存：

```sh
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
```

在终端下，执行以下命令，使环境变量立即生效：

```sh
source ~/.bashrc
```

Q: 如何修改 HuggingFace 缓存文件目录，以解决下载的模型文件占用系统盘空间过大，导致系统盘空间不足的问题？

A: 默认 HuggingFace 的缓存目录为 `/root/.cache`，可以通过以下命令永久修改 HuggingFace 的缓存目录：

在 `~/.bashrc` 文件的末尾，追加以下内容并保存：

```sh
export HF_HOME=/root/autodl-tmp/cache/hf
export HF_ENDPOINT=https://hf-mirror.com
```

在终端下，执行以下命令，使环境变量立即生效：

```sh
source ~/.bashrc
```

# 二、运行项目

```sh
cd /root/autodl-tmp
git clone https://github.com/yangyifeng1128/StableAvatar
cd StableAvatar
uv sync
mkdir checkpoints
uv run hf download FrancisRing/StableAvatar --local-dir ./checkpoints
```
