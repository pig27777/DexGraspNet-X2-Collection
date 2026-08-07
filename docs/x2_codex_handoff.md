# X2 5000-valid 采集的 Codex 账号交接

最后更新：2026-07-17 22:04 CST

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
- 当前 collector 若因 PTY/会话关闭而退出，已启用的用户级 supervisor 会在文件锁释放后
  按原参数自动恢复。生成阶段复用已提交 group；PhysX 阶段复用已审计的逐 JSON route，
  只丢失尚未原子落盘的当前小批。

## 可以继续使用的本地事实

- 工作区代码、`docs/*.md`、`data/x2_valid_5000` 和已写入的 attempt 文件都保留在本机。
- collector 有文件锁、严格 metadata 比较、group resume、PhysX route resume 和 completion hash
  审计；进程终止后可用同一输出根目录恢复。
- 新账号获得同一文件夹权限后，可以依据本文件和 attempt 证据继续，不需要依赖旧对话记忆。
- `x2-valid-collector-supervisor.service` 已链接到用户 systemd 并启用；新会话首先检查它和
  `collector_supervisor.log`，不再依赖旧 tool session ID。

## 当前权威快照

截至 2026-07-17 21:23 CST：

| 项目 | 状态 |
|---|---|
| 正式输出 | `data/x2_valid_5000` |
| 已完成 / 当前 attempt | `attempt_0000` / `attempt_0001` |
| `attempt_0001` raw target | 66757；f1--f5 = 16081/14247/11955/12746/11728；无 cap |
| catalog | 12 primitive + 30 general mesh，共 42 个物体 |
| catalog 文件审计 | 42/42 路径、scale、SHA 匹配；30 个通用 ID 唯一 |
| generator | v6，6000 iterations；正在生成 `attempt_0001` |
| validator | PhysX v7，六方向，100 logical steps，2 substeps，固定 batch 8 |
| 首轮完成结果 | 6250 raw = 642 valid + 5608 failed；42/42 物体；10.27% |
| 首轮 valid 分层 | front f1--f5 = 45/67/86/72/74；back = 62/53/57/63/63 |
| 首轮 pair / f5 | f1--f4 pair = 32/35/41/33；f5 front/back = 74/63 |
| 正在计算 | `attempt_0001`；seed 1009；首批 sphere 生成 worker |
| 故障策略 | 结构化 CUDA OOM 直接 fail-fast；不自动降 batch，不设 attempt raw cap |
| 当前 ETA | `attempt_0001` 证据边界 3.4--11.4 天，运行中心区间 6--9 天；若需新 attempt 则延长 |
| 后台守护 | `x2-valid-collector-supervisor.service` active + enabled |
| 最终审计 | 守护器会自动运行；仅 `final_audit.json passed=true` 且绑定 manifest SHA 后退出 |
| `complete.json` | `attempt_0000/complete.json passed=true` |
| `manifest.json` | 尚未生成 |
| 已审计候选池 / 最终完成 | 642 valid / 0 of 5000；尚无 manifest |

这个快照会过时。新会话必须运行下面的只读检查，以文件和进程的最新状态为准。
用户也可在仓库根目录长期运行不依赖 Codex 的只读仪表盘：

```bash
python3 scripts/watch_x2_collection.py
```

`Ctrl+C` 只退出仪表盘，不影响 collector 或 supervisor。

## 新账号接手后的第一组检查

```bash
cd /absolute/path/to/DexGraspNet-X2-Collection

systemctl --user status x2-valid-collector-supervisor.service --no-pager
tail -n 30 data/x2_valid_5000/collector_supervisor.log

pgrep -af 'collect_x2_valid_dataset.py|generate_x2_primitive_dataset.py|generate_x2_mesh_grasps_stratified.py'

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
  --validation-batch-size 8 \
  --sim-steps 100 \
  --general-mesh-root data/meshdata \
  --output-root data/x2_valid_5000 \
  2>&1 | tee -a data/x2_valid_5000/collector_console.log
```

不要修改 `attempt.json`，不要使用 `--overwrite`，不要删 `.collector.lock`、`.staging`、raw、
valid 或 failed。若命令报告 metadata/protocol/hash 不匹配，先保存错误并查明协议漂移，不能改
JSON 绕过。

正式参数固定 `--validation-batch-size 8`。wrapper 不会自动改成 4/2/1；出现结构化
`physx_batch_error`/CUDA OOM 时会硬失败并保留已原子路由的记录。collector 也没有 attempt raw cap，
下一 attempt 将用完成证明中的真实分层通过率计算补采量。

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

最终审计器实现不随公开仓库分发；其输出契约不变（退出码 0 且 `final_audit.json` 为
`passed=true`）。

只有命令退出码为 0 且 `final_audit.json` 为 `passed=true`，才可把 goal 标记 complete。

## 相关文档

- [断点恢复与运行日志](x2_collection_runbook.md)
- [实验与参数日志](x2_experiment_log.md)
- [正式数据协议](x2_primitive_dataset.md)
- [PhysX v7 验证协议](x2_physx_grasp_validation.md)
- [通用 mesh 生成器](x2_mesh_grasp_generator.md)
