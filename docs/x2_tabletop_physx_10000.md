# X2 混合物体桌面静态 + PhysX 10000 数据采集

入口：`scripts/collect_x2_tabletop_physx_dataset.py`

## 目标

- 最终恰好 10000 条 PhysX valid；
- 当前正式策略只采集 front/back × f2/f3/f5；每个面 f2=1667、f3=1667、f5=1666，
  合计恰好 10000 条；
- catalog 为 12 个 primitive + 正式 30 个通用 mesh；
- 通用 mesh 按最大边 90 mm 等比缩放，primitive 保留设计尺寸；
- 最终至少覆盖 30 个物体、其中至少 18 个通用 mesh，并覆盖 sphere、cylinder、cuboid、cube；
- 生成失败、桌面静态门失败、PhysX failed 和 PhysX valid 全部保留；
- 每条进入 PhysX 的姿态必须存在 FR5 数值 IK 解，并且装上 X2 后 FR5 的 6 个运动连杆与
  X2 碰撞几何都满足桌面净空；
- 支持 attempt 级断点续跑和单实例文件锁。

## 正式命令

```bash
cd /home/lhr/Desktop/DexGraspNet-main
set -o pipefail

mkdir -p data/x2_tabletop_physx_10000_f235_mu1

conda run -n isaaclab --no-capture-output \
  python scripts/collect_x2_tabletop_physx_dataset.py \
  --target-total 10000 \
  --finger-counts 2 3 5 \
  --output-root data/x2_tabletop_physx_10000_f235_mu1 \
  --n-iterations 6000 \
  --auto-gpu \
  --gpu-indices 0 \
  --generation-batch-size 32 \
  --sim-steps 100 \
  --substeps 2 \
  --hand-friction 1.0 \
  --object-friction 1.0 \
  --closing-contact-threshold 0.003 \
  --closing-displacement 0.002 \
  --closing-gradient-scale 100.0 \
  --closing-penetration-cap 0.0015 \
  --table-clearance-m 0.008 \
  --minimum-x2-root-table-distance-m 0.05 \
  --robot-table-clearance-m 0.005 \
  --fr5-object-table-xy-m 0.0 0.0 \
  --fr5-ik-seed-count 8 \
  --general-target-max-extent-m 0.09 \
  --expected-physx-pass-rate 0.10 \
  --minimum-raw-per-object-stratum 8 \
  --maximum-raw-per-object-stratum 32 \
  --minimum-object-coverage 30 \
  --minimum-general-coverage 18 \
  --resume \
  2>&1 | tee -a data/x2_tabletop_physx_10000_f235_mu1/collector.log
```

同一命令可直接恢复。已有进程仍运行时，文件锁会拒绝第二个 collector。

`--auto-gpu` 是默认模式。脚本通过 `nvidia-smi` 读取可见 GPU 和显存，并自动设置生成任务数、
PhysX 物体级并发数与 PhysX batch。当前 RTX 5090 32 GB 的解析结果为：2 个 batch-32 生成
worker，以及 2 个 batch-32 PhysX worker。`runtime_gpu_plan.json` 记录每次启动时实际采用的方案。
如需完全手动控制，可使用 `--manual-gpu --jobs N --validation-jobs N`。

## 输出布局

```text
data/x2_tabletop_physx_10000/
├── settings.json
├── progress.json
├── runtime_gpu_plan.json
├── collector.log
├── attempts/
│   └── attempt_NNNN/
│       ├── attempt.json
│       ├── complete.json
│       └── objects/<kind_object>/
│           ├── generation/f2,f3,f5/generator_output/
│           ├── static_failed/f2,f3,f5/<side>/
│           ├── physx_input/f2,f3,f5/<side>_single/
│           │   ├── raw/
│           │   ├── valid/
│           │   └── failed/
│           ├── physx.log
│           └── physx_summary.json
├── final/
│   ├── front/f2,f3,f5/
│   ├── back/f2,f3,f5/
│   └── manifest.json
└── manifest.json
```

`static_failed` JSON 带 `campaign_static_validation.reasons` 和
`fr5_mount_table_validation`。后者记录各 IK 初值、IK 误差、FR5 每个运动连杆的最小桌面净空和
碰撞连杆；典型失败原因包括 `FR5_IK_FAILED`、`FR5_TABLE_COLLISION`、
`X2_ROOT_TABLE_DISTANCE_FAILED`。PhysX failed JSON 保留六方向逐项结果和
`validation.failure_reasons`。最终 `manifest.json` 同时记录成功池、失败数量、物体覆盖、分层配额、
源路径和 SHA-256。

## 物理证据边界

物理验证使用仓库正式的 `x2_object_centered_dexgraspnet_six_orientation_v7`：单物体、无地面、
六重力方向 PhysX 抓持验证。桌面条件包含三道硬门：生成目标姿态下真实 X2 collision mesh
至少 8 mm 净空；X2 根离桌至少 50 mm；使用冻结的 FR5 桌面基座位姿和名义安装外参求 8 组
确定性 IK，并要求至少一个解让全部 FR5 运动连杆和带 IK 误差界的 X2 几何保持至少 5 mm
净空。FR5 链接检查使用官方 URDF collision mesh 对无限水平桌面求精确最低点；固定在桌面上的
`base_link` 按安装接口排除。无限平面比有限桌面更保守。

因此该数据证明“目标姿态存在 FR5+X2 整机不碰桌的静态安装构型 + 抓取通过六方向 PhysX”，
但不宣称已经完成
“物体在物理桌面上被手接近、闭合、抬离桌面”的动态 acquisition/lift 验证。
