FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY data/ ./data/
COPY models/ ./models/
COPY train.py .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p checkpoints logs

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令
CMD ["python", "train.py"]