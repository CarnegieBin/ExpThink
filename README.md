# ExpThink

基于 DAPO/GRPO 算法的大语言模型强化学习训练框架。

---

## 环境安装

### 1. 创建 conda 环境

```bash
conda create -n expthink python=3.10 -y
conda activate expthink
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意：** 版本不固定，核心依赖ray torch vllm兼容即可。

---

## 快速开始

> **必须在项目根目录 `ExpThink/` 下执行所有命令。**

```bash
bash src/run.sh
```

---

## 必须修改的参数

拿到代码后，请先修改 [src/run.sh](src/run.sh) 顶部的以下环境变量：

| 变量名 | 说明 | 示例 |
|---|---|---|
| `PROJECT_NAME` | 项目名称，用于日志和 checkpoint 目录命名 | `"ExpThink"` |
| `EXP_NAME` | 本次实验名称，区分不同实验 | `"deepseek_llm_1.5b"` |
| `MODEL_PATH` | 本地模型权重路径（HuggingFace 格式） | `"/ssd2/llm_models/DeepSeek-R1-Distill-Qwen-1.5B"` |
| `CKPTS_DIR` | checkpoint 保存根目录 | `"/home/work/tcbian/ExpThink/ckpts/..."` |

### 常用训练参数（按需调整）

| 参数 | 位置 | 说明 |
|---|---|---|
| `trainer.n_gpus_per_node` | run.sh | 每节点 GPU 数量，默认 `8` |
| `trainer.nnodes` | run.sh | 节点数，单机填 `1` |
| `trainer.total_epochs` | run.sh | 总训练轮次，默认 `10` |
| `trainer.test_freq` | run.sh | 每隔多少步做一次验证，默认 `10` |
| `trainer.save_freq` | run.sh | 每隔多少步保存一次 checkpoint，默认 `10` |
| `data.train_batch_size` | run.sh | 训练全局 batch size，默认 `512` |
| `data.gen_batch_size` | run.sh | 生成 batch size，默认 `256` |
| `actor_rollout_ref.actor.optim.lr` | run.sh | 学习率，默认 `1e-6` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | run.sh | vLLM 显存占用比例，默认 `0.85` |

---

## SwanLab 使用指南

本项目使用 [SwanLab](https://swanlab.cn) 进行实验追踪与可视化，训练脚本中已通过以下参数启用：

```bash
trainer.logger=['console','swanlab']
trainer.project_name="${PROJECT_NAME}"
trainer.experiment_name="${EXP_NAME}"
```

### 1. 安装 SwanLab

```bash
pip install swanlab
```

### 2. 登录（首次使用）

```bash
swanlab login
```

按提示输入你的 API Key（在 [swanlab.cn](https://swanlab.cn) 注册后获取）。

### 3. 查看实验

训练启动后，SwanLab 会自动上传日志。在浏览器打开 [https://swanlab.cn](https://swanlab.cn)，进入对应项目（`PROJECT_NAME`）即可查看：

- 训练/验证 loss 曲线
- reward 变化趋势
- 各测试集准确率
- 超参数记录

### 4. 离线模式（无公网环境）

如果服务器无法访问公网，可使用本地模式：

```bash
swanlab login --host http://<your-self-hosted-swanlab-server>
```

或在启动训练前设置环境变量关闭上传（仅本地记录）：

```bash
export SWANLAB_MODE=local
```

---

## 目录结构

```
ExpThink/
├── src/
│   ├── run.sh                  # 训练启动脚本（修改参数入口）
│   ├── main_dapo.py            # 训练主程序
│   └── custom_think_rm.py      # 自定义奖励函数
├── data/                       # 数据集目录（需自行准备）
└── ckpts/                      # checkpoint 保存目录（自动创建）
```
