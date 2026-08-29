# X2 左右模式与双物体候选数据集

## 模式定义

本项目的派生数据集固定使用以下映射：

| 原生成器字段 | 派生模式 |
|---|---|
| `active_side=front` | `right mode` |
| `active_side=back` | `left mode` |

原始 valid JSON 不被修改。`scripts/build_x2_dual_object_candidates.py` 通过硬链接将完成
attempt 中严格审计过的 valid 记录整理到：

```text
data/x2_dual_object/single_object/
├── right/f1 ... f5
└── left/f1 ... f5
```

`manifest.json` 同时保存 `mode` 和原始 `active_side`，避免命名转换造成歧义。

## 双物体组合

双物体只组合两个不同物体，并采用互补、无交集的手指集合：

```text
right f1 + left f4
right f2 + left f3
right f3 + left f2
right f4 + left f1
```

组合器按手指所有权合并 12 个 actuator：right 手指从 front valid 取值，left 手指从
back valid 取值；16 个 joint qpose 也按相同手指所有权合并。每条源记录中的 `hand_pose`
是物体坐标系下的手位姿，组合器取其逆变换，将两个物体都表达在同一个手坐标系下。

这里合并的是源 JSON 中的输入 qpose。已有 PhysX route 保存了 collision-aware closing 的
`selected_alpha` 和最大 actuator 调整量，但没有保存完整的最终 closing actuator target，
所以候选显式写入 `closing_actuator_target_saved=false`，不能声称恢复了验证末态 qpose。

这一步只产生 warm-start candidate。两条单物体记录各自通过 PhysX，并不能证明合并后的
共享关节状态能够同时抓住两个物体。因此候选明确写入：

```json
"dual_object_validation": {
  "status": "not_run"
}
```

在联合验证完成前，不得将这些候选称为 dual-object valid。

## 生成命令

仅使用具有 `complete.json` 完成证明的 attempt：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/build_x2_dual_object_candidates.py \
  --output-root data/x2_dual_object
```

小规模试验可限制每种互补组合的数量：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/build_x2_dual_object_candidates.py \
  --pairs-per-combination 20 \
  --output-root data/x2_dual_object
```

若输出目录已经存在，命令默认拒绝覆盖；只有显式传入 `--overwrite` 才会重建。

采集尚未完成时，可以显式把当前已经严格通过的 route 做成中间快照：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/build_x2_dual_object_candidates.py \
  --include-incomplete-attempts \
  --output-root data/x2_dual_object_snapshot
```

这种输出的 `formal_source_completion_required=false`，只用于提前开展联合验证；最终正式派生
数据仍应在 attempt completion proof 写出后重建。

## 联合验证要求

双物体正式 valid 至少需要重新证明：

1. 合并 actuator 后的共享手 FK 正确且关节不越限；
2. 手与两个物体分别满足穿透阈值；
3. 两个物体之间没有不允许的初始碰撞；
4. 六个重力方向中两个物体均保持手部接触；
5. 两个物体均满足位移、有限状态、mimic 误差和 actuator tracking 阈值。

联合验证必须使用两个独立 rigid object actor，且接触判据必须逐物体计算，不能把任意一个
物体的接触当作两个物体都通过。

## 自动联合 PhysX

联合验证器为 `scripts/validate_x2_dual_object_physx.py`，协议 revision 为
`x2_dual_object_six_orientation_physx_v1`。它先重新计算合成 qpose 的 sampled 手部自碰撞、
两次手物穿透和物物穿透；静态门通过后，在共享手坐标系中同时生成两个 dynamic rigid
object。物物 PhysX collision 会在静态无穿透证明后禁用，因此每个 object contact sensor
报告的末态接触只能直接来自手，不能由另一物体托住。

每条候选测试六个重力方向、每方向 100 logical steps × 2 substeps。只有两个物体在六方向均
保持直接手部接触、位移小于 0.1 m、状态 finite，且合成手的 active joint tracking 与 Newton
mimic 误差分别小于 0.1/0.01 rad，才写入 `physx_validation/valid/`；其余原子路由到
`physx_validation/failed/`。原始 warm-start JSON 不修改，`--resume` 会复核 source SHA 和协议
后跳过已有 route。

手动命令：

```bash
conda run -n isaaclab --no-capture-output \
  python scripts/validate_x2_dual_object_physx.py \
  --dataset-root data/x2_dual_object \
  --output-root data/x2_dual_object/physx_validation \
  --batch-size 4 --sim-steps 100 --substeps 2 \
  --resume --headless --device cuda:0
```

用户级 `x2-dual-object-physx-after-completion.service` 已启用。它等待正式
`data/x2_valid_5000/manifest.json` 证明 5000 valid，随后从 completed attempts 重建不同物体
组合。四类组合各至少 500 条，但不再截断在 500：所有能够满足不同物体和互补手指约束的候选
都会进入联合 PhysX。服务失败时保留所有已原子路由结果并由 systemd 重启续验；最终
`physx_validation/summary.json` 必须绑定当时 composition manifest SHA-256，并证明
`valid + failed = manifest 中的全部候选数`。

```bash
systemctl --user status x2-dual-object-physx-after-completion.service --no-pager
journalctl --user -u x2-dual-object-physx-after-completion.service -f
```
