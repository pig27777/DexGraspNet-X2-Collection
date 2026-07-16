# X2 5000-valid 采集的 Codex 账号交接

最后更新：2026-07-16 22:50 CST

## 给新 Codex 会话的接手提示

切换账号后，在 Codex 中打开同一仓库根目录（下文记为 `<repo_root>`），然后原样发送下面这段话：

```text
请创建并持续执行下面这个 goal，不要把 pilot、raw 或未完成 attempt 当成完成：

持续监控并运行 <repo_root> 的正式 X2 数据采集，直到
data/x2_valid_5000/manifest.json 严格证明恰好 5000 条 Isaac Sim/PhysX v7 valid。
正式 catalog 必须是 12 个 primitive + 固定 30 个通用 mesh；front/back 各 2500，
每侧 f1、f2、f3、f4、f5 各 500。front f1↔back f4、f2↔back f3、
f3↔back f2、f4↔back f1 必须同物体且 finger set 不重叠；f5 为单侧记录。

开始操作前请完整阅读：
1. docs/x2_codex_handoff.md
2. docs/x2_collection_runbook.md
3. docs/x2_experiment_log.md
4. docs/x2_primitive_dataset.md
5. docs/x2_physx_grasp_validation.md

先用进程、attempt 文件和 manifest 检查当前真实状态。如果旧 collector 仍活着，只监控，
不要启动第二个；如果已退出，严格使用 x2_collection_runbook.md 中的同参数恢复命令。
每个 completed attempt 后更新两份日志。只有最终 manifest 的逐项审计全部通过后，才能把
goal 标记 complete。
```

## 不能自动迁移的内容

- 旧账号的 Codex 对话、active goal 状态、tool session ID 和对话内存不能假设会出现在新账号。
- 当前 collector 的 Codex PTY session ID 是旧会话内部状态，新会话不要依赖该编号。
- 当前 collector 若因旧 PTY 关闭而退出，已启用的用户级 supervisor 会在文件锁释放后按原参数
  自动恢复；未到原子提交点的最多两个物体仍需重算。

## 可以继续使用的本地事实

- 工作区代码、`docs/*.md`、`data/x2_valid_5000` 和已写入的 attempt 文件都保留在本机。
- collector 有文件锁、严格 metadata 比较、group resume、PhysX route resume 和 completion hash
  审计；进程终止后可用同一输出根目录恢复。
- 新账号获得同一文件夹权限后，可以依据本文件和 attempt 证据继续，不需要依赖旧对话记忆。
- `x2-valid-collector-supervisor.service` 已链接到用户 systemd 并启用；新会话首先检查它和
  `collector_supervisor.log`，不再依赖旧 tool session ID。

## 当前权威快照

截至 2026-07-16 23:47 CST：

| 项目 | 状态 |
|---|---|
| 正式输出 | `data/x2_valid_5000` |
| 当前 attempt | `attempt_0000` |
| raw target | 6250；f1--f5 各 1250 |
| catalog | 12 primitive + 30 general mesh，共 42 个物体 |
| catalog 文件审计 | 42/42 路径、scale、SHA 匹配；30 个通用 ID 唯一 |
| generator | v6，6000 iterations，2 CUDA workers |
| validator | PhysX v7，六方向，100 logical steps，2 substeps |
| 已完成生成对象 | 12 个 primitive + 通用 mesh `000`、`003` |
| 正在计算 | 通用 mesh `006`、`009` |
| 已发布 raw/valid/failed | 2100 / 0 / 0 |
| 分层 | front/back × f1--f5 每格 210 |
| raw 静态审计 | 最近完整审计的 primitive 子集：1800 finite/self-collision；1775 dense feasible，25 明确静态失败；新增通用 raw 待全量阶段审计 |
| 互补 | 已生成 840 pair；primitive 子集 720 pair 中 696 对双方 dense feasible；仍待 PhysX |
| 后台守护 | `x2-valid-collector-supervisor.service` active + enabled |
| 最终审计 | 守护器会自动运行；仅 `final_audit.json passed=true` 且绑定 manifest SHA 后退出 |
| `complete.json` | 尚未生成 |
| `manifest.json` | 尚未生成 |
| 正式完成度 | 0/5000；不能用运行时间代替已验证记录 |

这个快照会过时。新会话必须运行下面的只读检查，以文件和进程的最新状态为准。

## 新账号接手后的第一组检查

```bash
cd /absolute/path/to/DexGraspNet-X2-Collection

systemctl --user status x2-valid-collector-supervisor.service --no-pager
tail -n 30 data/x2_valid_5000/collector_supervisor.log

pgrep -af 'collect_x2_valid_dataset.py|generate_x2_primitive_dataset.py|generate_x2_mesh_grasps_stratified.py|validate_x2_primitive_dataset.py|validate_x2_mesh_grasps_physx.py'

nvidia-smi

find data/x2_valid_5000/attempts -path '*/raw/*.json' -type f | wc -l
find data/x2_valid_5000/attempts -path '*/valid/*.json' -type f | wc -l
find data/x2_valid_5000/attempts -path '*/failed/*.json' -type f | wc -l
find data/x2_valid_5000/attempts -name complete.json -type f -print

test -f data/x2_valid_5000/manifest.json && \
  jq '{passed, valid_count, side_finger_counts,
       covered_general_object_count, paired_entry_count,
       single_side_five_finger_entry_count}' \
  data/x2_valid_5000/manifest.json
```

判断规则：

1. 只要旧顶层 collector 或生成/验证 child 仍在运行，就不要启动重复任务。
2. 进程全部不存在且没有 manifest 时，按
   [断点恢复手册](x2_collection_runbook.md) 的完整同参数命令恢复。
3. 只有 `.staging` 而没有 raw 时，当前内存物体没有形成恢复点；恢复时会以相同 seed 重跑。
4. raw 不是 valid；partial valid 也不是正式完成计数。只有 audited `complete.json` 的 attempt
   才进入候选池。
5. 即使候选池超过 5000，也必须等最终 `manifest.json` 证明精确分层、互补配对和 30-mesh 覆盖。

## 当前正式恢复命令

只有在上述 `pgrep` 确认旧进程全部退出后才能执行：

```bash
cd /absolute/path/to/DexGraspNet-X2-Collection
set -o pipefail

conda run -n isaaclab --no-capture-output \
  python scripts/collect_x2_valid_dataset.py \
  --target-valid 5000 \
  --n-iterations 6000 \
  --generation-device cuda \
  --jobs 2 \
  --validation-device cuda:0 \
  --validation-batch-size 32 \
  --sim-steps 100 \
  --general-mesh-root data/meshdata \
  --output-root data/x2_valid_5000 \
  2>&1 | tee -a data/x2_valid_5000/collector_console.log
```

不要修改 `attempt.json`，不要使用 `--overwrite`，不要删 `.collector.lock`、`.staging`、raw、
valid 或 failed。若命令报告 metadata/protocol/hash 不匹配，先保存错误并查明协议漂移，不能改
JSON 绕过。

## 完成标准

新账号只能在当前文件系统中的 `data/x2_valid_5000/manifest.json` 同时证明以下条件后宣布完成：

- `passed=true`、`valid_count=5000`；
- front/back 各 2500；每侧 f1--f5 各 500；
- 2000 个互补双侧 pair，即 4000 条配对记录；
- 1000 条 f5 单侧记录；
- 每个 pair 同物体且 front/back finger set 不相交；
- 30 个固定通用 mesh 全部覆盖；
- 全部记录来自 v6 raw 和 PhysX v7 六方向通过结果；
- manifest 引用的所有 attempt `complete.json` 路径及 SHA-256 可重新验证。

最后还必须运行：

```bash
set -o pipefail
conda run -n isaaclab --no-capture-output \
  python scripts/audit_x2_valid_dataset.py \
  --output-root data/x2_valid_5000 \
  --general-mesh-root data/meshdata \
  | tee data/x2_valid_5000/final_audit.json
```

只有命令退出码为 0 且 `final_audit.json` 为 `passed=true`，才可把 goal 标记 complete。

## 相关文档

- [断点恢复与运行日志](x2_collection_runbook.md)
- [实验与参数日志](x2_experiment_log.md)
- [正式数据协议](x2_primitive_dataset.md)
- [PhysX v7 验证协议](x2_physx_grasp_validation.md)
- [通用 mesh 生成器](x2_mesh_grasp_generator.md)
