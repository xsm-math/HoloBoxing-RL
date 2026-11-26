# 本机实测记录

2026 年 11 月 26 日在 GTX 1660 Ti（6 GiB 显存）、Ubuntu 24.04 WSL2 环境复测。WSL2 使用 7.5 GiB 主存与 2 GiB 交换空间。运行使用 Python 3.11、PyTorch 2.9.1 与 CUDA 12.8。 [复现命令](../docs/最小仿真复现.md)

| 阶段 | 实测进度 | 阶段总耗时 | 整卡显存采样峰值 |
| --- | --- | ---: | ---: |
| 数据采集 | 8 个片段、11,462 条样本，完整性检查通过 | 408.86 s | 2,550 MiB |
| 监督模仿 | 1 轮，使用 430 条 InterHuman 训练样本，4 个优化步 | 49.01 s | 204 MiB |
| 站立预热 PPO | 4 环境 × 8 步，1 次更新，32 条转移 | 184.94 s | 2,010 MiB |

成功运行的 PPO 阶段包括冻结底层校验和仿真初始化。单次更新用时 5.17 s，策略优化器与 AMP 判别器各执行两步。 [机器可读训练记录](local-smoke.json)

高层 ONNX 为 2,359,996 字节。官方导出脚本在六组输入上比较 PyTorch 与 ONNX，最大绝对误差为约 `5.66e-7`。v12 本地评测通过单文件格式、尺寸和接口检查，并用 CUDA 完成一次推理。完整对战和完整链路时延待测。 [官方本地校验报告](v12-validation.json)

## 本机内存与运行图

首次直接启动 PPO 在生成批量 ONNX 后触发了 WSL 主存 OOM。复测成功时进程常驻内存峰值为 6,621.9 MiB。采集、监督模仿与 PPO 开始前设置 glibc 内存分配参数。

```bash
export GLIBC_TUNABLES=glibc.malloc.arena_max=2:glibc.malloc.mmap_threshold=131072:glibc.malloc.trim_threshold=131072
```

复测在独立进程调用官方 `ordered_topk.execution_source()`，将 4 环境执行图保存到文件。辅助脚本在训练进程加载图前检查原图、转换源码和缓存图的 SHA-256，官方底层权重的 SHA-256 与锁定清单一致。 [运行图缓存脚本](../work/local-smoke/runtime_graph_cache.py) · [官方冻结模型清单](https://github.com/HorizonRobotics/HoloMotion/blob/f52dd81ae935a828b0296f5ee0f833beb7378bf0/holoboxing/evaluation/frozen-model.json)

仓库提供 [CUDA 探测](../work/local-smoke/gpu_probe.py) 和 [采样监控](../work/local-smoke/monitor.py) 脚本。训练检查点、ONNX 和源数据由复现流程在本地生成或按资源许可获取。
