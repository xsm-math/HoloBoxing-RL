# HoloBoxing-RL

复现地平线 H-Bot 赛题二的官方仿真训练流程，围绕 PPO、GAE、AMP、实测状态模仿、自博弈和分层控制开展源码学习。

## 可用于对战的模型

新增 [联赛最终交付模型](models/league-final/README.md)：`reactive_r3`，18 场 15 胜 3 负、零摔倒；对队友执红 62∶0、执蓝 105∶0。包含最终 ONNX，以及队友、两款历史防御版、初始进攻基线和其他进攻型对手的完整对战摘要。

新增 [主动得分对战基线](models/scoring-baseline/README.md)，含自包含 ONNX、模型哈希和本地对战结果，供防守策略研究。该模型对队友提交版的标准红蓝对局结果为 77∶0、78∶0，原有训练复现记录见下文。

## 当前进展

官方 HoloMotion 以 Git 子模块固定在 `f52dd81ae935a828b0296f5ee0f833beb7378bf0`。训练沿用官方 MuJoCo 环境、PPO、AMP 与冻结的 HoloMotion 2.0 底层控制器。 [官方中文版教程](https://github.com/HorizonRobotics/HoloMotion/blob/f52dd81ae935a828b0296f5ee0f833beb7378bf0/holoboxing/README.zh-CN.md)

初步复现已完成单进程采集、动作先验、监督模仿和 4 环境的一次站立预热 PPO 更新。高层 ONNX 通过官方 v12 的接口与单次 CUDA 推理校验。完整仿真对战和上下层推理链时延待测。 [实测记录](results/README.md)

| 从这里开始 | 内容 |
| --- | --- |
| [训练架构与源码导读](docs/训练架构与源码导读.md) | 模型与仿真数据流、损失公式、关键路径和函数 |
| [最小仿真复现](docs/最小仿真复现.md) | 依赖与数据准备、训练、导出、校验和下一轮对战路线 |
| [rl-smoke.json](configs/rl-smoke.json) | 兼容官方 `TrainConfig` 的小规模配置 |
| [实测结果](results/README.md) | 训练进度、资源峰值、导出一致性和 v12 校验范围 |

## 获取源码

```bash
git clone https://github.com/xsm-math/HoloBoxing-RL.git
cd HoloBoxing-RL
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 upstream/HoloMotion
```

资源、底层模型和训练数据的获取方式见[复现步骤](docs/最小仿真复现.md#1-linux-源码与依赖)。本地下载、权重和训练中间产物放在 `work/`，导出与运行日志放在 `outputs/`。本机的 WSL 虚拟磁盘存放在项目的 `work/wsl/Ubuntu-24.04/` 下。
