# X2 Primitive 初始抓取数据集

本数据集为当前 X2 通用 mesh 抓取生成器提供第一版、单位为米的确定性 primitive 输入。
它只增加 mesh 构建和批量调度层，不修改优化器、能量、contact sampling、X2 手模型或官方
DexGraspNet 源码。

下文已经落盘的 500/500-step 数值表是修复自碰撞前的 v3 历史数据，不满足 v4
`self_collision.feasible` 契约。使用当前生成器重建时，每条 raw 还必须包含 capsule/hull 能量、
静态自碰撞诊断和 checkpoint provenance；当前 PhysX 路由会把静态 gate 加入 overall success，
但 `simulation_success` 仍只表示物理抓取与 mimic 结果。

## 正式数据协议

官方 DexGraspNet 2.0 的 **88 个通用 mesh** 全部由
`scripts/select_x2_general_meshes.py` 审计，并在
`data/meshdata/x2_general_mesh_manifest.json` 中逐物体记录 SHA-256 和
`object_scale=1.0`；不能用目录里临时混入的其他 mesh 补足数量。当前正式数据从这 88 个中按
catalog 顺序确定性选取 **30 个**（ID 为 `000,003,...,087`），并与 12 个 primitive 组成
42-object 正式 catalog。这样满足通用物体提升到 20--30 个的目标，同时保留完整 88-object
inventory 审计。

正式完成条件是严格的 **5000 条 Isaac Sim/PhysX valid**：front/back 各 2500 条，且每侧
f1、f2、f3、f4、f5 各 500 条。同物体的 front/back 只按互补手指数配对，两侧参与手指集合
必须互斥；f5 因为已经使用全部五指，只保留单侧条目。raw 数量、能量下降或某个未完成 attempt
都不是完成证据，最终以 `data/x2_valid_5000/manifest.json` 及其引用的 attempt
`complete.json` 为准。

## Primitive catalog

共 12 个 watertight OBJ：

```text
data/meshdata/x2_primitives/
├── sphere/
│   ├── sphere_r020.obj
│   ├── sphere_r030.obj
│   └── sphere_r040.obj
├── cylinder/
│   ├── cylinder_r018_h100.obj
│   ├── cylinder_r025_h100.obj
│   └── cylinder_r032_h080.obj
├── cuboid/
│   ├── cuboid_x035_y055_z090.obj
│   ├── cuboid_x045_y065_z110.obj
│   └── cuboid_x055_y075_z130.obj
└── cube/
    ├── cube_e040.obj
    ├── cube_e050.obj
    └── cube_e060.obj
```

文件名尺寸标签使用三位毫米整数，OBJ 顶点仍以米保存。Sphere 使用固定 subdivision 的
icosphere，cylinder 固定沿 local Z 轴并使用 64 sections，cuboid/cube 使用中心在原点的
triangle box。构建后重新加载 OBJ，并检查有限顶点、合法非退化三角形、watertight、winding、
正体积、中心和精确尺寸。

构建或重新审计：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/build_x2_primitive_dataset.py --overwrite
```

不传 `--overwrite` 时，已有 OBJ 不会重写，但仍会完整审计。

## 批量生成逻辑

`scripts/generate_x2_primitive_dataset.py` 默认可对每个 `(instance, side, finger_count)` 调用
现有单 mesh 生成器。正式闭环使用 `--stratified-batching`：同一物体尚缺的 front/back、f1..f5
行由一个常驻手模型的 row-policy 进程生成，避免为每个分层重复加载模型；输出仍按 side/finger
严格分组并分别从 sample index 0 开始。

固定参数：

| 参数 | 值 |
|---|---:|
| 普通子进程 batch size | 8 |
| stratified resident batch size | 64 |
| contact count | f1..f4 为 4；f5 为 5 |
| surface samples | 512 |
| primitive object scale | 1.0 |
| 正式通用 mesh object scale | manifest 中的 1.0 |

批量 CLI 默认 `num_grasps=64`、`n_iterations=6000`、`seed=0`。500 轮只用于 smoke/对比组，
正式样本默认与 DexGraspNet 源码保持同一数量级。CLI 支持：

```text
--shapes sphere cylinder cuboid cube
--side front|back|both
--finger-counts 1 2 3 4 5
--complementary-side-fingers
--num-grasps N
--finger-targets N1 N2 N3 N4 N5
--n-iterations N
--device DEVICE
--seed N
--jobs N
--target-total N
--overwrite|--resume
--stratified-batching
--include-general-meshes
--general-mesh-root PATH
--general-mesh-ids ID [ID ...]
```

`--jobs` 并行运行互不共享状态的 instance-side 子进程；默认 1。RTX 5090 实测
`--jobs 2` 将本任务 GPU 利用率由约 47% 提升到约 95%，显存约 8.5 GB。`--target-total`
与 `--num-grasps` 互斥，用于精确指定整个所选 catalog 的总样本数；余数按固定 catalog 顺序
分配，因此结果可审计和复现。

`--finger-targets` 与 `--finger-counts` 一一对应，用于给自适应 attempt 指定各手指数的精确
raw 总量。`--stratified-batching` 只允许和 `--resume`、显式 `--finger-counts`、
`--complementary-side-fingers` 一起使用。resume 会严格复核已发布组的 mesh、scale、seed、迭代数、
样本连续编号和 finger mask；只复用完整组，缺失或损坏组重新生成。

`--mesh-root` 和 `--output-root` 额外用于隔离测试或自定义数据根目录。

现有单 mesh 生成器固定写入 `<side>_single/raw`。Primitive 批量层使用同一文件系统中的临时
staging，完整验证数量、连续索引、active side、mesh path、12 actuator、16 joint、4 个唯一
contact ID（f5 为 5 个）、有限能量和未验证状态后，再原字节发布到：

```text
data/x2_primitive_grasps/<shape>/<side>/raw/
```

示例文件名：

```text
sphere_r020_front_000000.json
cylinder_r018_h100_back_000007.json
cuboid_x035_y055_z090_front_000063.json
cube_e040_back_000000.json
```

`--overwrite` 只清理当前 instance/side 前缀，避免 64 条之后运行 8 条时遗留旧尾部，同时不会
删除其他 instance 或 side。脚本不创建 `valid` 或 `failed`。

## CSV summary

每次成功运行原子写入：

```text
data/x2_primitive_grasps/summary.csv
```

列固定为：

```text
shape
size
side
finger_count
sample_count
finite_sample_count
mean_initial_energy
mean_final_energy
energy_decreased_count
maximum_penetration_mean
maximum_penetration_min
maximum_penetration_median
```

所有统计都从最终发布的 raw JSON 内容重新计算，不依赖单 mesh 生成器的 stdout 汇总。

## Smoke 与历史 raw 候选命令

每实例每侧 8 条、500 步 smoke，总计 `12 × 2 × 8 = 192` 条：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/generate_x2_primitive_dataset.py \
  --shapes sphere cylinder cuboid cube \
  --side both \
  --num-grasps 8 \
  --n-iterations 500 \
  --device cuda \
  --seed 0 \
  --overwrite
```

当前实际 smoke 结果：24 个 instance-side run 全部完成，写入 192 个 raw JSON；
`finite=192/192`、能量下降 `192/192`，CSV 为 24 行。按 shape/side 汇总如下（penetration
为 mean/min/median，单位 m）：

| Shape | Side | 样本数 | 平均能量 initial → final | Maximum penetration mean/min/median |
|---|---|---:|---:|---:|
| sphere | front | 24 | `210.682327 → 13.507015` | `0.000529079 / 0 / 0` |
| sphere | back | 24 | `208.344217 → 16.131782` | `0.002476862 / 0 / 0.000230992` |
| cylinder | front | 24 | `199.661182 → 16.573231` | `0.000885676 / 0 / 0.000407166` |
| cylinder | back | 24 | `197.821275 → 16.216991` | `0.001220710 / 0 / 0.000005374` |
| cuboid | front | 24 | `194.547873 → 19.022398` | `0.003108771 / 0 / 0.003527758` |
| cuboid | back | 24 | `193.592395 → 17.297230` | `0.002130666 / 0 / 0.001748844` |
| cube | front | 24 | `204.944010 → 18.124692` | `0.000956964 / 0 / 0.000021707` |
| cube | back | 24 | `205.439207 → 17.185487` | `0.001239444 / 0 / 0.000693854` |

全体平均能量为 `201.879061 → 16.757353`；maximum penetration 全体
mean/min/median 为 `0.001568522 / 0 / 0.000533751 m`。没有创建 `valid`、`failed`、
`front_single` 或 `back_single` 目录。

早期 primitive-only 6000-step raw 基线总计 `12 × 2 × 64 = 1536` 条，写入独立目录以保留
500 轮对比组。它不是当前 5000-valid 正式数据集：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/generate_x2_primitive_dataset.py \
  --shapes sphere cylinder cuboid cube \
  --side both \
  --num-grasps 64 \
  --n-iterations 6000 \
  --device cuda \
  --seed 0 \
  --output-root data/x2_primitive_grasps_6000 \
  --overwrite
```

早期还可扩充为精确 5000 条 raw 并使用两个并发 GPU worker：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/generate_x2_primitive_dataset.py \
  --shapes sphere cylinder cuboid cube \
  --side both \
  --target-total 5000 \
  --n-iterations 6000 \
  --device cuda \
  --jobs 2 \
  --seed 0 \
  --output-root data/x2_primitive_grasps_6000 \
  --overwrite
```

24 个 instance-side group 中，固定顺序的前 8 组各生成 209 条，其余 16 组各生成 208 条；
front/back 各 2500 条，总计恰好 5000。

如需把仓库中 `<object>/coacd/decomposed.obj` 通用物体纳入生成，增加
`--include-general-meshes`；可再用 `--general-mesh-ids` 限定 ID。官方 DexGraspNet 2.0 mesh
包的 **88 个**通用物体全部经过确定性筛选和审计；正式采集仅使用其中固定的 30 个 ID，另保留
12 个 primitive。完整 inventory 的筛选清单、SHA-256、实际尺度、类别和拒绝原因记录在
`data/meshdata/x2_general_mesh_manifest.json`。筛选入口：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/select_x2_general_meshes.py \
  --source-root /tmp/dexgraspnet2_meshdata_extracted \
  --destination data/meshdata \
  --target-count 88 \
  --object-scale 1.0 \
  --replace-existing
```

`--replace-existing` 先只从 source 严格审计并选满 88 个，再把未选旧对象安全移动到
`data/meshdata/_excluded_general_meshes/<id>`，不会删除旧资产。完成后直系
`data/meshdata/*/coacd/decomposed.obj` 必须恰好 88 个。审计以 `object_scale=1.0` 后的物理尺寸
为准；正式生成也从 manifest 读取同一个尺度。当前 30-object 正式子集为：

```text
000 003 006 009 012 015 018 021 024 027
030 033 036 039 042 045 048 051 054 057
060 063 066 069 072 075 078 081 084 087
```

primitive 同样使用米制尺度 `1.0`。通用物体输出文件名前缀使用物体 ID，避免多个
`decomposed.obj` 相互覆盖。

## 5000 valid、双侧五种手指数与互补配对

正式终止条件是 **5000 条 Isaac Sim/PhysX valid**，不是 5000 条 raw。最终配额为：

- front：f1/f2/f3/f4/f5 各 500 条，共 2500 valid；
- back：f1/f2/f3/f4/f5 各 500 条，共 2500 valid；
- `finger_participation` 中的 `target_count`、`actual_count` 和 `finger_names` 必须与
  `selected_contacts` 一致；palm 不计入手指数；
- 同一物体按 `front f1 ↔ back f4`、`front f2 ↔ back f3`、`front f3 ↔ back f2`、
  `front f4 ↔ back f1` 合并，两侧 `finger_names` 必须不相交；
- f5 已使用全部五指，不存在非空且不重叠的另一侧集合，因此 f5 保留为单侧 valid，
  manifest 中 `pair_id=null`，不伪造双侧配对。

批量生成使用 `--complementary-side-fingers`，同一物体的上述 front/back 组合从初始化到
退火 contact 重采样始终保持互补。f1–f4 使用 4 个唯一 contact，f5 使用 5 个唯一 contact；
PhysX validator 按 contact 数拆 batch，避免混合 4/5-contact ragged tensor。

闭环控制器会执行“生成 → PhysX 验证 → 审计已完成 attempt → 统计每个 side/finger 层和可配对
数量 → 换 seed 补采”，直至达到全部配额，再将恰好 5000 条 valid 硬链接到
`final_valid/` 并写 manifest：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/collect_x2_valid_dataset.py \
  --target-valid 5000 \
  --n-iterations 6000 \
  --generation-device cuda --jobs 2 \
  --validation-device cuda:0 --validation-batch-size 32 --sim-steps 100 \
  --general-mesh-root data/meshdata \
  --output-root data/x2_valid_5000
```

首次没有历史有效率时，每个尚缺 finger stratum 至少申请 500 条 raw，并按缺口乘 1.25
预留；后续 attempt 使用已完成 attempt 的实际 PhysX 有效率做保守估计。这里的
`--minimum-attempt-raw` 是 raw 下限，不是 valid 配额，不能据此提前结束。

每个 attempt 的 `attempt.json` 固定 seed、迭代数、finger raw targets、12+30 物体目录及
hash/scale、验证协议和 100 步参数。生成阶段使用 resume 与 stratified batching；验证阶段也使用
`--resume`，跳过已经存在且仍匹配 raw 的 valid/failed 路由结果。wrapper 对 primitive 使用
`convex-hull`，对通用非凸 mesh 使用 `convex-decomposition`。

attempt 只有同时满足以下条件才会原子写入 `complete.json` 并进入统计：

- `summary.csv` 行数、finger 分层和 raw 总数与 `attempt.json` 完全一致；
- `generation_summary.json` 证明 resume、64-row stratified batching、每个物体/side/finger 组、
  6000 轮配置及逐物体 scale 都与 attempt 元数据一致；
- `validation_summary.csv` 覆盖全部 42 个物体（12 primitive + 30 general），且每条 raw 恰好
  路由到一个 valid 或 failed；
- `complete.json` 保存 attempt metadata、生成 CSV、生成 JSON、验证 summary 的 SHA-256 以及
  raw/valid/failed 数量；每次读取都会重新计算，陈旧证明立即拒绝。

进程中断后，collector 先恢复没有 `complete.json` 的 attempt：复用严格审计通过的生成组和
已有 PhysX 路由，补齐缺失部分；未完成 attempt 在证明生成前不会贡献任何 valid 计数。

最终 `manifest.json` 是完成证据，必须同时满足 `valid_count=5000`、每个 side/finger=500、
所有非空 pair 同物体且两侧手指集合不相交、验证后端/协议为正式 PhysX v7、每条记录都有
100 个逻辑仿真步和六个全部通过的方向，并且最终选择覆盖全部 30 个正式通用物体。manifest 还保存
5000 条来源与 SHA-256、2000 个双侧 pair（共 4000 条配对记录）、1000 个单侧 f5 条目，以及所有被采用 attempt 的
`complete.json` 路径和 SHA-256。最终抽样按物体轮询，避免按文件名截断造成后部物体没有进入
数据集。

现有 `data/x2_primitive_grasps` 是从旧的 5000 条、500 轮运行中确定性保留的 500 条对比组，
不是正式数据：front/back 各 250 条，24 个 instance-side group 均有覆盖。500/500 能量下降且
finite，但 4 个接触距离之和 `E_dis` 的 mean/median/min 分别为
`0.0760293 / 0.0717904 / 0.0189646 m`，没有一条低于 `0.005 m`，因此不能据此宣称已形成抓取。
其总能量均值为 `201.3159 → 17.5604`，maximum penetration mean/median/min 为
`0.00159981 / 0.000393218 / 0 m`。正式 6000 轮样本必须写入上面的独立目录并经过 PhysX validator。

所有输出都是 `success=false`、`validation.status=not_run` 的 raw candidates。能量下降或 sampled
penetration 不能替代 Isaac Sim/PhysX 抓取有效性验证。

数据采集完成后使用独立 X2 validator；生成期间不与单卡 GPU 争用。完整协议见
[`x2_physx_grasp_validation.md`](x2_physx_grasp_validation.md)，primitive 批量入口为：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/validate_x2_primitive_dataset.py \
  --input-root data/x2_primitive_grasps \
  --shapes sphere cylinder cuboid cube \
  --side both \
  --batch-size 32 \
  --sim-steps 100 \
  --device cuda:0 \
  --resume
```

当前正式采集状态、参数标定、正负实验结果与证据路径持续记录在
[X2 抓取数据采集实验日志](x2_experiment_log.md)；中断、断电、验证失败后的同参数续采步骤见
[正式采集运行日志与断点恢复手册](x2_collection_runbook.md)。
