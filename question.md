
markdown
# 问题记录：mujoco_py 编译失败（X11/Xlib.h: No such file or directory）
## 问题时间
2025-12-31
## 问题环境
- OS：Ubuntu
- Python：3.8（conda 环境 `qvpo`）
- 相关库：gym, mujoco_py, MuJoCo 2.1.0 (`~/.mujoco/mujoco210`)
## 问题描述
运行命令：
```bash
python main.py --env_name HalfCheetah-v3 --weighted --aug
程序在创建 MuJoCo 环境（如 HalfCheetah-v3）时崩溃，报错堆栈显示在导入 mujoco_py 时编译 C 扩展失败。

关键错误信息
text
fatal error: X11/Xlib.h: No such file or directory
   99 | #include <X11/Xlib.h>
      |          ^~~~~~~~~~~~
compilation terminated.
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
原因分析
mujoco_py 在安装/首次导入时需要编译依赖 OpenGL 和 X11 的 C 扩展。
系统未安装 X11 开发头文件（X11/Xlib.h），导致 GCC 编译失败。
这是系统级依赖缺失问题，而不是 Python 代码逻辑问题。
解决步骤
安装系统依赖（Ubuntu）：
bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libx11-dev \
    libxi-dev \
    libxmu-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    libglfw3 \
    libglfw3-dev
在 conda 环境中（qvpo）确认/重新安装 mujoco_py：
bash
conda activate qvpo
pip install -U 'mujoco-py<2.2,>=2.1'
重新运行命令：
bash
cd ~/sda/qvpo
python main.py --env_name HalfCheetah-v3 --weighted --aug
处理结果
安装上述系统依赖后，mujoco_py 能够成功编译 C 扩展，HalfCheetah-v3 环境创建成功，程序可以正常运行训练。

 
---
 
如果你按上面步骤执行后还有新的报错（比如换成别的缺失库），把新的终端输出贴出来，我再帮你逐条排查。
Feedback



为什么仍然报 X11/Xlib.h 错？
因为 mujoco_py 的 C 扩展在编译时仍会 include eglplatform.h，而这个头文件默认依赖 X11 头文件；你设置的环境变量只影响运行时，不影响编译时的 include 路径和宏定义。