# X2 Isaac Sim / PhysX 抓取验证

## 1. 边界

X2 使用独立 validator：

```text
grasp_generation/x2_isaac_validation.py
scripts/validate_x2_mesh_grasps_physx.py
scripts/validate_x2_primitive_dataset.py
```

它不修改、导入或调用官方 DexGraspNet 的 ShadowHand Isaac Gym validator。官方入口只接受
ShadowHand 22-DOF `.npy`，不能用于 X2 schema-1 JSON。

生成器仍然只写 `raw`。validator 不修改 raw 原文件，而是把包含物理结果的副本原子写到同级
`valid` 或 `failed`：

```text
<shape>/<side>/raw/<sample>.json
<shape>/<side>/valid/<sample>.json
<shape>/<side>/failed/<sample>.json
```

官方 DexGraspNet 2.0 的 88 个通用 mesh 全部受 inventory manifest 审计；正式采集对
12 个 primitive 和其中确定性选出的 30 个通用 mesh 使用本 validator。
最终 5000 条数据中的每一条都必须是 `validation.status=passed`、`success=true`、
`simulation_success=true` 的 PhysX 路由副本；raw 或只通过 schema dry-run 的记录不计入配额。

## 2. 重放坐标

生成 JSON 使用：

```text
p_object = R_hand @ p_x2_root + t_hand
```

其中物体位于 object/world 原点。validator 与生成器使用完全相同的物体中心坐标：物体在每个
environment origin 保持单位姿态，X2 根节点直接写入 JSON 的 `R_hand, t_hand`：

```text
R_object = I
t_object = 0
R_x2 = R_hand
t_x2 = t_hand
```

JSON 四元数是 `wxyz`，当前 Isaac Lab root-state API 是 `xyzw`；转换在进入仿真前显式完成。
front/back 共用同一个 X2 USD、FK 和关节状态，back 不做镜像。

每批开始前还会用 JSON 中被选 contact 的 `link_name`、`local_position`、
`local_surface_normal` 对 Isaac runtime FK 做点和法向审计。这个审计失败属于 validator/frame
错误，命令直接非零退出，不能把样本误写成 `failed`。

## 3. 12 actuator 与 16 joint

初态按 JSON `joint_names` 写入全部 16 个 runtime DOF；目标只按名字写给 12 个真实主动关节。
四个 J1 follower 继续由 USD 的 `NewtonMimicAPI` 约束驱动，不创建第二套 actuator：

```text
LFJ1 = LFJ2
RFJ1 = RFJ2
MFJ1 = MFJ2
FFJ1 = FFJ2
```

手的 effort/velocity limit 和显式低顶点 PhysX collision hull 来自
`x2_mujoco/x2_keypoints.usda`。正式 v7 稳定性标定在 runtime 固定并逐条记录：active drive
stiffness=`1000 N·m/rad`、damping=`0.632455532 N·m·s/rad`、armature=`0.0001 kg·m²`；
使用 TGS（`solver_type=1`）、每步在每个 TGS position iteration 施加外力，并保持
`solve_articulation_contact_last=false`。这组参数是 X2/Isaac Sim 6 的数值稳定性标定，不照搬
ShadowHand 的统一 drive 参数。validator 会审计实际 runtime drive、solver 与命令元数据，任一
不一致都不能进入正式 manifest。

X2 USD 当前显式设置 `newton:selfCollisionEnabled=0` 和
`physxArticulation:enabledSelfCollisions=0`。本 validator 保持该动力学设置，不把 PhysX 当作
手指自碰撞 oracle；link-link 判定来自生成器与 PhysX 共用低顶点 collision hull 的确定性
双向表面采样。本问题涉及的手指 visual mesh 最多可能比该低顶点 hull 额外伸出约
`0.65 mm`（精确逐 link 值见诊断 CLI），因此静态门槛不是原始 visual mesh 的连续无交证明。

当前 Isaac Sim 6 / PhysX runtime 会直接消费 `NewtonMimicAPI`；旧的
`PhysxMimicJointAPI` 已弃用，因此 validator 不叠加第二套 mimic，也不给 J1 添加 drive。运行时会
审计 actuator ownership，并在每个物理步累计四组 `abs(J1-J2)`；任何一组超过 `0.01 rad` 都被
视为该样本对应方向的物理跟踪失败，记录 `newton_mimic_tracking:<orientation>`。它不会再让同一
GPU batch 中其他无关样本中止；summary 同时报告越限 orientation/sample 数量和全局最大误差。

## 4. 六方向协议

默认 `dexgraspnet-contact` 口径参考官方源码：每条候选测试六个完整配置方向，**每个方向各
仿真 100 个逻辑步**、`dt=1/60 s`，每步使用 2 个 PhysX substep；不是六个方向合计 100 步。
六个方向末态都仍有 hand-object contact 才令 `simulation_success=true`。因此正式证据是
`100 logical steps × 6 orientations`（每方向 200 个 physics substeps）。

六个共同旋转对应 object/config frame 中的重力：

```text
-Y, +Y, -X, +X, +Z, -Z
```

validator 在全局零重力场景中向各物体质心施加 object frame 下的逐环境等效重力。这里不再乘
候选自己的 `R_hand.T`，因为手和物体的相对姿态已与生成器 JSON 完全一致。这样可在一个 batch 内并行运行
`batch_size × 6` 个隔离环境，且与官方“共同旋转手和物体、保持世界重力不变”在数学上等价。

PhysX GPU tensor backend 不支持把 articulation link 用作 filtered contact partner。validator
因此不使用净合力或 articulation filter，而调用 Object contact view 的
`get_raw_contact_data()`：末帧 raw contact patch count 大于 0 即表示存在接触。每个隔离环境严格
只有一个 Object、一个 X2 Robot、没有地面，且跨环境 collision 已过滤，所以 Object 的任意对方
actor 都只能是同环境 X2。这个判据不会因相向手指的力互相抵消而产生假阴性；诊断接触力取该
Object 所有 raw patch 中的最大法向力。

最终成功定义为：

```text
simulation_success = 六个方向全部 finite、末态存在 X2-object contact pair，且 Newton mimic 未越限
penetration_pass = raw.maximum_penetration < 0.001 m
self_collision_pass = v6 raw.self_collision.feasible
success = simulation_success && penetration_pass && self_collision_pass
```

`simulation_success` 始终只表示 PhysX 抓取/mimic 结果，不受静态自碰撞 gate 影响。精确 v4+
pipeline 必须提供一致的 `self_collision` 诊断；v3 或缺失该字段的
旧记录继续采用旧 success 语义。这里的两个 penetration 都是生成器 sampled diagnostic，不是
连续碰撞证书。validator 同时记录物体
最终/最大位移、全过程最大主动关节跟踪误差和接触力。可选 `strict-hold` 会把位移与关节误差也
加入通过条件。

当前正式协议名为 `x2_object_centered_dexgraspnet_six_orientation_v7`：六方向和“末态仍有接触”的判据参考
DexGraspNet 源码，但不是逐参数复制。默认会执行一次源码式 contact-gradient closing：初态仍是
raw JSON 的 16 joint，目标只调整 12 个 active actuator，4 个 J1 继续由 Newton mimic 驱动；
最近表面点来自 X2 真实 collision surface，并补充当前抓取选择的 authored keypoint。X2 的梯度
尺度标定为 100（ShadowHand 源码为 500），near-surface 参与范围标定为 3 mm；后者只决定哪些
link 参加预闭合，不放宽末态 PhysX contact 判据。可用 `--no-force-closing` 关闭。验证使用 X2 自身的
USD solver/drive/collision、9.8 m/s² 重力和 sampled maximum penetration。因此结果应解释为 X2
适配验证，不应与 ShadowHand 官方结果逐数值比较。

## 5. 物体 collision asset

OBJ 在 AppLauncher 启动后由 Isaac Lab `MeshConverter` 转成带刚体、质量和 collision 的缓存
USD。cache key 包含 OBJ 内容、scale、collision approximation、density、contact/rest offset、
Isaac Sim/Isaac Lab 版本及 validator protocol revision。
转换后必须核对 USD bounds 与 scaled OBJ bounds，防止单位或轴向被静默改变。

- primitive 数据集全部为凸体，wrapper 固定使用 `convex-hull`；
- 一般 mesh CLI 默认使用 `convex-decomposition`；
- 正式 30 个通用 mesh 的 inventory manifest 必须记录 `object_scale=1.0`，每条 raw 的
  `object.scale` 也必须为 `1.0`；wrapper 会复核候选尺度，USD cache key 也包含该尺度；
- 同一 mesh 的正式分层数据可同时包含 4-contact（f1..f4）与 5-contact（f5）记录；
  validator 先按 contact 数分组，再按 `--batch-size` 切分，禁止把 ragged contact ID
  直接构造成同一个 tensor；
- density 默认 `500 kg/m³`；
- hand/object friction 默认 `3.0`；
- X2 contact offset 默认 `0.001 m`；没有照搬 ShadowHand validator 的 `0.01 m`，因为对本数据集中
  直径 40 mm 的最小球体而言，10 mm 接触壳会把明显间隙计作 speculative contact；
- 场景不创建地面；Object raw contact 的对方 actor 因而只能是同环境 X2 hand。

validator 进程在加载 NumPy/SciPy 前把 OpenBLAS、OMP、MKL 和 NumExpr 线程数固定为 1，避免
Kit 启动阶段 fork 与 BLAS worker pool 冲突。这个设置只影响 validator 的 CPU 数值辅助代码；
PhysX 环境仍在 GPU 上批量运行。

## 6. 命令

X2 articulation 拓扑、12 个 active actuator 的 ownership、四组 Newton follower、runtime FK
和接触法向审计已经内置在 PhysX validator 中。任何审计失败都会直接非零退出，且不会把候选误写为
`failed`，无需额外运行独立 articulation probe。

单个 mesh 的 schema 与路由 dry-run（不启动 Isaac Sim）：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/validate_x2_mesh_grasps_physx.py \
  --input-root data/x2_primitive_grasps \
  --mesh-path data/meshdata/x2_primitives/sphere/sphere_r020.obj \
  --side both \
  --dry-run \
  --device cuda:0
```

单个 mesh 的物理验证：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/validate_x2_mesh_grasps_physx.py \
  --input-root data/x2_primitive_grasps \
  --mesh-path data/meshdata/x2_primitives/sphere/sphere_r020.obj \
  --side both \
  --batch-size 32 \
  --sim-steps 100 \
  --criterion dexgraspnet-contact \
  --collision-approximation convex-hull \
  --device cuda:0 \
  --viz none \
  --resume
```

交互查看一条已通过候选时，使用同一套 v7 drive、TGS 和六方向验证代码；命令先完成六方向
重放，再把 identity 环境的最终状态冻结在 Isaac Sim 窗口中，关闭窗口即退出：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/view_x2_mesh_grasp_physx.py \
  --raw-json /tmp/x2_sphere_valid_line_v6_n0/front_single/raw/sphere_r020_front_000031.json \
  --state validated-final \
  --collision-approximation convex-hull \
  --sim-steps 100 --substeps 2 \
  --device cuda:0 --viz kit --max_visible_envs 6
```

完整 primitive 数据集在采集结束后顺序验证：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/validate_x2_primitive_dataset.py \
  --input-root data/x2_primitive_grasps \
  --shapes sphere cylinder cuboid cube \
  --side both \
  --batch-size 32 \
  --sim-steps 100 \
  --criterion dexgraspnet-contact \
  --device cuda:0 \
  --resume
```

正式 attempt 同时验证 12 个 primitive 和 30 个通用 mesh；闭环 collector 会自动构造等价于
下面的 wrapper 调用：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/validate_x2_primitive_dataset.py \
  --input-root data/x2_valid_5000/attempts/attempt_0000 \
  --shapes sphere cylinder cuboid cube \
  --include-general-meshes \
  --general-mesh-root data/meshdata \
  --side both \
  --batch-size 32 \
  --sim-steps 100 \
  --criterion dexgraspnet-contact \
  --device cuda:0 \
  --resume
```

`--resume` 跳过已经存在 valid/failed 副本的 raw；`--overwrite` 重新验证并保证同一文件不会同时
残留在 valid 和 failed。两者互斥。collector 重启时会先恢复未完成 attempt；已有路由仍需通过
raw source 路径和 SHA-256 审计，不能用陈旧副本冒充完成结果。

## 7. 输出审计

每个结果保留原 contact/energy/optimization 内容，只更新：

```text
success
simulation_success
validation.status
validation.backend = isaac_sim_physx
validation.protocol_revision
validation.criterion
validation.source_sha256
validation.thresholds
validation.preflight
validation.orientations[6]
validation.failure_reasons
validation.runtime
```

loader 会逐项核对 `selected_contact_ids/selected_contacts`、contact 向量与单位法向、side 支持、
12→16 mimic 映射和 raw 所在 side 目录。对 v4 还会验证 `self_collision.maximum_penetration`、
`total_penetration`、`threshold` 均有限且非负，`worst_pair` 形状合法，并要求 `feasible` 与
`maximum_penetration <= threshold` 一致。`validation.preflight` 明确记录静态 gate 是否必需及
是否通过；失败原因使用 `self_collision_not_feasible`。加载时固定 raw bytes/SHA-256，发布前再次逐字节比较；
验证期间 raw 若变化就拒绝写结果。仿真发生 NaN/Inf 时测量值写为 JSON `null`、样本路由 failed，
不会用非法浮点破坏整个 batch。

序列化继续使用 `allow_nan=False` 和临时文件原子 rename。primitive wrapper 另外生成
`validation_summary.csv` 及逐物体 JSON summary；每个子进程使用唯一临时 summary，经 mesh、
scale、数量和输出路由复核后才原子发布。

对正式闭环，`validation_summary.csv` 本身还不是 attempt 完成证明。collector 只有在生成
`summary.csv`、`generation_summary.json` 与验证 summary 同时覆盖 `12+30=42` 个物体、全部
raw 恰好路由一次、finger 分层、64-row stratified batching 和逐物体 scale 都与
`attempt.json` 一致后，才写入 `attempts/attempt_NNNN/complete.json`。该文件包含 attempt
metadata、生成 CSV/JSON、验证 summary 的 SHA-256 以及 raw/valid/failed 数量；后续统计会
重新计算这些字段，缺失或陈旧的 `complete.json` 所属 attempt 完全不计数。

最终 `data/x2_valid_5000/manifest.json` 还会逐条复核正式后端/协议、恰好 100 个逻辑步、六个
方向名称和六方向全部 passed，并证明 front/back 各 2500、每侧 f1..f5 各 500、同物体互斥
手指配对、f5 单侧以及全部 30 个正式通用 mesh 的覆盖。只有这份 manifest 和其中引用的
`complete.json` hash 全部成立，才能称为 5000 valid 已完成。

## 8. 历史 v3 PhysX 基线与 v4 A/B

修复前 v3 使用 `sphere_r020.obj`、front、32 条、6000 轮候选，以协议 v5 默认参数运行
32×6 个 PhysX environment，结果为 `1 valid / 31 failed`。按旧 success 语义的首条 valid 为：

```text
data/x2_formal_grasps_6000/sphere_r020_seed1/front_single/valid/
  sphere_r020_front_000031.json
```

该文件的 raw SHA-256 与验证前字节完全一致；raw 仍保留在 `raw/`，同名 `failed` 不存在。关键
审计值：

- total energy：`217.448817 → 1.063887`；
- `E_dis=0.00320354 m`，sampled maximum penetration=`0.0000373630 m < 0.001 m`；
- 4 个唯一 contact ID：`palm:front:040`、`finger:middle:001`、`palm:front:038`、
  `finger:middle:002`；
- object-centered runtime FK 最大 contact point error=`1.35e-6 m`，最小 normal dot=`0.99999994`；
- 六方向均 finite 且末态存在接触，final contact force 范围=`0.4225–0.5906 N`；
- 六方向 maximum displacement 范围=`0.000685–0.000792 m`；
- 六方向最大 Newton mimic error=`0.004602 rad < 0.01 rad`；
- v3 `success=true`、`simulation_success=true`、failure reasons 为空。

该 raw 的 SHA-256 为
`d4b6bd97826490e7356b3c536997c8390fe3ee9ca12f7eb09eb20e38a160b489`，但静态 hull
复核发现 thumb-index 最大穿透约 `8.08 mm`；因此它只是 v3 物理基线，不能作为 v4 valid。

v4 A/B 必须保持相同 mesh/side/seed/batch/iterations 和 64×3 静态采样，并至少满足：32/32
self-collision 不超过 `0.5 mm`，至少 31/32 hand-object penetration 小于 `1 mm`，PhysX 成功
不少于旧基线 1/32，有限 orientation 不少于 140/192，mimic violation sample 不多于 26。
报告应分别给出上述主样本和整批结果，且明确 PhysX self-collision 仍为 disabled。

2026-07-15 对固定 v3 raw 与本次 v4 raw 运行同一 protocol v5 validator 的实际结果：

| 指标 | v3 before | v4 after | 目标与结论 |
|---|---:|---:|---|
| static self-collision feasible | 8/32（离线 64×3 复核；validator gate 0/32） | 32/32（gate 32/32） | 32/32，通过 |
| hand-object penetration pass | 31/32 | 30/32 | 至少 31/32，**未通过** |
| `simulation_success` | 1/32 | 1/32 | 纯 PhysX 口径不下降，通过 |
| overall `success` | 1/32（无 self gate） | 1/32（含 self gate） | 数量相同，定义不同 |
| finite orientations | 140/192 | 142/192 | 至少 140，通过 |
| nonfinite-or-mimic runtime-mask orientations | 55 | 52 | 改善 3 个 orientation |
| nonfinite-or-mimic runtime-mask samples | 26 | 29 | 不多于 26，**未通过** |
| 其中有限 mimic `>0.01 rad` orientations / samples | 3 / 3 | 2 / 2 | 有限跟踪越限改善 |
| 最大有限 mimic error | `1.8221e18 rad` | `0.0336599 rad` | 大幅改善，但仍有越限 |

runtime mask 把非有限 orientation 的 mimic error 视为无限大：before 有 52 个 nonfinite
orientation、涉及 24 条样本；after 为 50 个、涉及 28 条样本。它们分别与有限 mimic 越限样本
重叠 1 条，因此 sample union 为 26 和 29。29 的回归主要来自 nonfinite 分散到更多样本，不应
误读为有限 mimic tracking 本身恶化。

主样本 31 在修复前后均保持 `simulation_success=true` 且六方向 finite；其最大 Newton mimic error
从 `0.00460184 rad` 变为 `0.00568289 rad`，仍低于 `0.01 rad`。v4 主样本六方向最大物体
位移范围为 `0.0008995–0.0038853 m`。运行时审计确认两次 A/B 的 PhysX articulation
self-collision 都是 disabled；v3 离线复核与 v4 静态 gate 使用同一生成器双向 sampled
collision-hull oracle。

因此本轮证明了自碰撞门槛和 PhysX 抓取成功数没有回归，但还不能宣称完整正式验收通过：
hand-object 门槛少 1 条、nonfinite-or-mimic runtime-mask sample 多 3 条。这两个剩余问题没有通过修改 mimic、
动力学或 validator 语义来掩盖，应在后续候选质量调优中单独处理。

当前 v7 参数标定、逐实验动态结果、正式 pilot 与 5000 条采集进度持续记录在
[X2 抓取数据采集实验日志](x2_experiment_log.md)。
