# X2 正式数据采集运行日志与断点恢复手册

最后更新：2026-07-27 10:53 CST

## 技术摘要

- 正式 collector 使用 `data/x2_valid_5000` 作为唯一输出根目录；重复执行同一条命令会先恢复
  未完成 attempt，不会从新的 attempt 跳过旧工作。
- 生成阶段按“物体内全部待生成手指数层”运行，并在严格审计后逐 JSON 原子发布；只有文件名
  连续、数量完整且 provenance 全部匹配的 object/side/finger group 才会在恢复时复用。中断时
  仍在内存中或只发布了一部分文件的物体需要以相同 seed 重新跑。
- PhysX v7 按 batch 验证，每条 `valid/failed` JSON 原子写入。恢复时会复核已有路由、删除与
  当前协议不一致的陈旧路由，只计算尚未路由的 raw。
- `attempt.json`、生成 summary、验证 summary、`complete.json`、最终 `manifest.json` 构成
  逐级完成证据。只有 `complete.json` 通过重新哈希审计的 attempt 才贡献正式候选；只有最终
  `manifest.json` 证明恰好 5000 条及全部配额时，整个采集才完成。
- 这里的“续训”实际是续采集。当前生成器是 6000 轮 simulated annealing，不是神经网络训练；
  它没有单个物体内部逐 iteration checkpoint。恢复粒度是已提交 group，最坏只重跑中断时
  正在运行的最多两个物体，而不是重跑全部 42 个物体。
- 用户级 `x2-valid-collector-supervisor.service` 已启用。它通过同一个内核文件锁观察 collector，
  不会启动重复任务；锁释放且最终 manifest 未完成时，自动执行完全相同的正式恢复命令。

## 当前正式运行快照

| 项目 | 当前值 |
|---|---|
| 首次启动 / 最近恢复 | 2026-07-16 17:06 / 2026-07-26 14:21:07 CST |
| 输出根目录 | `data/x2_valid_5000` |
| 已完成 / 当前 attempt | `attempt_0000` / `attempt_0001` |
| `attempt_0001` raw target | 66757；f1--f5 为 16081/14247/11955/12746/11728；非全局 cap |
| catalog | 12 primitive + 固定 30 general mesh |
| 生成协议 | `x2_mesh_grasp_unselected_finger_side_v6`，6000 iterations |
| 验证协议 | `x2_object_centered_dexgraspnet_six_orientation_v7` |
| 最终物化协议 | `x2_balanced_cross_object_complementary_30mesh_5000_v7`；运行中 attempt 元数据仍为 v6 |
| 21:19 状态 | `attempt_0000` 完成；`attempt_0001` 已自动开始生成 |
| 已完成 raw/valid/failed | 6250/642/5608；42/42 物体；`complete.json passed=true` |
| 首轮 valid 分层 | front f1--f5 = 45/67/86/72/74；back = 62/53/57/63/63 |
| 首轮互补 / f5 | pair f1--f4 = 32/35/41/33；f5 front/back = 74/63 |
| 验证执行策略 | 固定 `batch=8`；明确 OOM fail-fast；不自动降 batch，不设 attempt raw cap |
| 已审计 completed 候选池 / 当前 passed route | 642 / 7061；最终仍为 0 of 5000，尚无 `manifest.json` |
| 2026-07-27 10:58 生成进度 | attempt 0001 为 63589/66757 raw；剩余 `084/087` 仍在内存生成 |
| 提前 PhysX | 前 38 个物体已路由 60421 条：6419 valid + 54002 failed |
| 跨物体热加载 | `x2-cross-object-pairing-reload.service` 等 generation summary 后一次性重启 supervisor |

当前没有趋势图：只有一个 completed attempt，样本不足以形成稳定趋势。partial route 可用于
诊断，但不能作为正式吞吐或完成证明。每个 attempt 完成后再追加同协议下的
raw/valid/failed 与 side/finger_count 表格。

## 跨会话后台守护

守护器实现不随公开仓库分发；持久用户单元为
`systemd/x2-valid-collector-supervisor.service`。该单元已链接并启用，Codex 账号或终端切换不会
依赖旧对话的 PTY；若操作系统重启，它会在该 Linux 用户下次登录时恢复采集。

```bash
systemctl --user status x2-valid-collector-supervisor.service --no-pager
tail -f data/x2_valid_5000/collector_supervisor.log
```

守护器每 15 秒探测 `.collector.lock`，每 5 分钟把已发布 raw/valid/failed 写入持久日志。它只在
锁空闲且 manifest 没有严格证明 5000 条配额时恢复；manifest 的 headline 配额满足后会自动
运行最终独立审计（实现不随公开仓库分发）。只有全量审计报告与当前 manifest SHA-256 绑定且通过，服务才
发布 `final_audit.json` 并退出；失败报告保存为 `final_audit_failed.json`，不能提前结束 goal。

双物体派生数据另由
`x2-dual-object-physx-after-completion.service` 等待正式 collection manifest。它不会在
采集/生成期间占用 GPU；正式 5000-valid 完成后才重建四类各 500 条跨物体组合并执行联合六方向
PhysX。状态和日志：

```bash
systemctl --user status x2-dual-object-physx-after-completion.service --no-pager
journalctl --user -u x2-dual-object-physx-after-completion.service -f
```

### 独立终端仪表盘

不需要保持 Codex 对话打开。在仓库根目录新开一个终端执行：

```bash
python3 scripts/watch_x2_collection.py
```

仪表盘默认每 5 秒刷新，显示：

- 最终 manifest / independent audit 状态；
- 已审计 valid 候选池、f1--f4 互补 pair 和 front/back f5 进度；
- 每个 attempt 的 raw target、raw/valid/failed、阶段和 42-object 验证进度；
- supervisor/collector/generator/validator PID、耗时、CPU、RSS、当前物体；
- GPU 显存、利用率、温度、功耗和最近结构化事件。

`Ctrl+C` 只关闭仪表盘，不会停止 collector。它不获取文件锁、不写数据、不发送
任何进程信号。其他常用模式：

```bash
# 只打印一次，适合 SSH 或脚本检查
python3 scripts/watch_x2_collection.py --once

# 每 10 秒刷新，不清屏，便于保留终端历史
python3 scripts/watch_x2_collection.py --interval 10 --no-clear
```

若要**主动暂停**，必须先阻止守护器重启任务：

```bash
touch data/x2_valid_5000/.stop_supervisor
systemctl --user stop x2-valid-collector-supervisor.service
pgrep -af 'collect_x2_valid_dataset.py|generate_x2_mesh_grasps_stratified.py'
```

若 `pgrep` 仍显示顶层 collector，再对其 PID 发送一次 `SIGINT` 并等待全部 child 退出。
重新续采时删除 sentinel 并启动服务：

```bash
rm -f data/x2_valid_5000/.stop_supervisor
systemctl --user start x2-valid-collector-supervisor.service
```

## 文件状态就是恢复点

| 文件或目录 | 含义 | 中断后处理 |
|---|---|---|
| `.collector.lock` | 内核文件锁的载体；文件存在不等于进程仍持锁 | 不删除；重新运行时由 collector 尝试加锁 |
| `attempts/attempt_NNNN/attempt.json` | seed、6250 raw 配额、42 个物体、v6/v7 参数契约 | 必须保留且禁止手工修改 |
| `attempt_NNNN/**/raw/*.json` | 已审计 group 的逐文件原子发布结果 | 恢复时按连续索引、数量、seed、迭代数、finger mask、scale、dense gate 复核后复用 |
| `attempt_NNNN/.staging/` | 当前仍在生成、尚未提交的临时物体 | 不计数；旧进程退出后由下一次 collector 自动清理 |
| `summary.csv` + `generation_summary.json` | 该 attempt 全部 raw 已生成并通过全量生成审计 | 两者都存在且匹配时跳过生成阶段 |
| `**/valid/*.json`、`**/failed/*.json` | 已原子完成的 v7 路由 | 恢复时逐条审计并跳过；陈旧路由自动删除重跑 |
| `validation_summaries/*.json` | 每个物体的完整路由报告 | 物体完成后原子替换；不完整临时 summary 不发布 |
| `validation_summary.csv` | 42 个物体全部路由完成 | 存在且审计通过时跳过验证阶段 |
| `complete.json` | raw/valid/failed 数量和四类 summary/metadata SHA-256 的 attempt 证明 | 每次读取都重算；缺失时该 attempt 贡献 0 条正式数据 |
| `manifest.json` | 恰好 5000 valid 的最终证明 | 只有它通过全量复核才可宣布完成 |

## 中断后五步恢复

### 1. 先确认旧进程是否仍活着

在仓库根目录执行：

```bash
cd /absolute/path/to/DexGraspNet-X2-Collection

pgrep -af 'collect_x2_valid_dataset.py|generate_x2_primitive_dataset.py|generate_x2_mesh_grasps_stratified.py'

systemctl --user status x2-valid-collector-supervisor.service --no-pager
nvidia-smi
df -h .
```

如果仍有顶层 `collect_x2_valid_dataset.py`，不要启动第二个 collector，也不要单独启动 validator。
collector 自身还有非阻塞文件锁，重复进程会以 `Another collector already holds ...` 退出。

如果要主动暂停，先按“跨会话后台守护”一节写入 stop sentinel 并停止 supervisor，再向顶层
collector 发送一次 `Ctrl-C`，然后等待上述
`pgrep` 不再显示任何 collector/generator/validator。不要只杀掉一个生成 child，因为仍存活的
顶层进程可能立即调度下一物体；如果顶层已死但存在 orphan child，应对 `pgrep` 显示的明确 PID
发送 `SIGINT` 并确认退出，再执行恢复命令。

### 2. 旧进程确实退出后，保留所有数据

不要删除或编辑以下内容：

- `data/x2_valid_5000/attempts/`
- `.collector.lock`
- `.staging/`
- 任意 `raw/`、`valid/`、`failed/`
- `attempt.json`、`complete.json`、`manifest.json`

`.collector.lock` 的实际锁在进程退出时由内核释放，留下的空文件不妨碍恢复；`.staging` 会由
collector 在审计已提交 group 后自动清理。手工删除可能把可诊断证据一起删掉。

### 3. 用完全相同的正式命令重新运行

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

`tee -a` 让恢复运行拥有持久控制台日志；`set -o pipefail` 保证 collector 失败时整条管道也
返回失败。当前 supervisor 恢复进程已写入 `collector_console.log`；权威状态仍以 attempt
文件为准，不应为了日志主动停止采集。

正常恢复开头应出现类似：

```text
[collector] resuming attempt_0000
[resume] reusable_groups=<已提交层数> regenerate_groups=<待生成层数>
```

若复杂通用 mesh 触发 CUDA OOM，正式 wrapper 会立即输出明确根因并失败，不自动降低 batch。
已经完成的 route 仍会原子保留，查明并修复后可继续 `--resume`。
为让问题尽快暴露，正式路径不做 8→4→2→1 自动回退，也不为下一 attempt 添加人为数量上限。

不要添加 `--overwrite`，也不要改 seed、6000 iterations、100 sim steps、30-mesh 列表或输出根目录。
恢复命令若报告 `Attempt metadata changed`，禁止修改 `attempt.json` 来绕过；应先恢复与原 attempt
匹配的代码/参数，或把新协议放到新的输出根目录，不能把两个协议混进同一 attempt。

### 4. 观察恢复是否真正复用了数据

```bash
# 已原子提交的 raw；不等于正式 valid
find data/x2_valid_5000/attempts/attempt_0000 \
  -path '*/raw/*.json' -type f | wc -l

# 已路由结果；验证中断后这两类会继续增长
find data/x2_valid_5000/attempts/attempt_0000 \
  -path '*/valid/*.json' -type f | wc -l
find data/x2_valid_5000/attempts/attempt_0000 \
  -path '*/failed/*.json' -type f | wc -l

# 完成证明和最终证明
find data/x2_valid_5000/attempts -name complete.json -type f -print
test -f data/x2_valid_5000/manifest.json && \
  jq '{passed, valid_count, side_finger_counts, covered_general_object_count}' \
  data/x2_valid_5000/manifest.json
```

对生成恢复，`reusable_groups` 应大于或等于中断前已发布层数。对验证恢复，每个物体报告中的
`skipped_existing_count` 应反映已有 valid/failed；若已有路由与当前 v7 或 raw SHA 不符，wrapper
会明确打印 `[resume] removed stale route ...` 并只重跑这些记录。

### 5. 只在完成证明存在后更新正式计数

```bash
find data/x2_valid_5000/attempts -name complete.json -type f -print0 | \
  xargs -0 -r jq -s \
  '{completed_attempts:length,
    raw:(map(.raw_count)|add // 0),
    valid:(map(.valid_count)|add // 0),
    failed:(map(.failed_count)|add // 0)}'
```

这个累计 valid 仍只是可供最终配对选择的池，不等于最终 5000。正式结束必须由
`manifest.json` 同时证明每侧 f1--f5 各 500、2000 个互补双侧 pair、1000 个 f5 单侧条目和
30 个通用 mesh 全覆盖。front/back pair 可以来自不同物体，但手指集合必须互补且不相交；
每对另有一份 `final_pairs/` 组合记录，分别保留两侧物体与 qpose。

manifest 出现后必须运行独立只读审计器；退出码为 0 且报告 `passed=true` 才能结束 goal：

最终审计器实现不随公开仓库分发；其输出契约保持不变：退出码为 0 且报告 `passed=true`
才发布 `final_audit.json`。

该审计器不信任 manifest 的 headline：它重新哈希全部 5000 个 final 文件、验证 hard link 与
source、一一重跑 v6/v7 JSON 契约检查、重算 attempt completion proofs、连续分层索引、2000 个
跨物体允许的互补 pair 及其 `final_pairs/` 组合文件、1000 个 f5 单侧条目以及固定 30-mesh 覆盖。当前 manifest 尚不存在时，它按
预期返回退出码 1 和 `final manifest is missing`，不会把 partial 数据误判完成。

## 不同中断位置会损失多少计算

| 中断位置 | 已保存 | 需要重跑 | 不会重跑 |
|---|---|---|---|
| 单个物体的 6000 轮生成中 | 之前已提交的物体/group | 当前仍在内存中的物体 | 之前已提交 group、其他 completed attempt |
| group 提交过程中 | 已完成原子 rename 的 JSON | 未完整通过 group 审计的部分 | 完整连续索引且 provenance 匹配的 group |
| 全部 raw 后、summary 前 | 所有已发布 raw | summary 重建及任何严格审计失败的 group | 审计通过的 raw |
| PhysX 一个 batch 中 | 前面已原子写出的 valid/failed | 当前内存 batch | 前面已路由且 v7/SHA 匹配的记录 |
| 一个物体验证后、总 CSV 前 | 逐条路由和已发布物体 summary | 缺失 summary/未路由物体 | 已审计路由 |
| `complete.json` 写入前 | metadata、raw、路由和 summaries | completion proof 重建 | 全部通过哈希审计的内容 |
| 最终 materialize 中 | completed attempts | `final_valid/` 和 manifest 的确定性重建 | 原始 attempt 数据 |

## 常见故障处理

### 终端关闭、SSH/IDE 断开或机器重启

确认旧进程不存在后直接执行同一恢复命令。机器重启会释放文件锁；PhysX USD cache 和已提交
attempt 数据仍可复用。

### CUDA OOM、驱动错误或非有限值

先确认 GPU 上没有残留生成/验证进程，再用 `nvidia-smi` 检查显存。不要未复现就自动降 batch、
修改 drive 或放宽 v7 门槛。先保存错误日志并按同参数复现；确认 batch 32 和 16 均在
`012` OOM、batch 8 可完整路由后，正式并行数才统一锁定为 8。后续再出现 OOM 会直接失败，
不由 wrapper 自动改参。

### 磁盘满

只清理与正式 attempt 无关的缓存或 `/tmp` pilot，保留 `data/x2_valid_5000`。释放空间后用同一
命令恢复。不要为了空间删除 failed：`complete.json` 要求 `valid + failed = raw`，且最终证明会
重新校验数量。

### metadata、summary 或哈希不匹配

这是保护性失败，不是可以忽略的 warning。不要手改 JSON/CSV 或复制文件凑数。保存完整错误，
比对 `attempt.json`、代码协议常量、mesh selection manifest 与 contact-candidate SHA；只有恢复
一致协议或重新生成受影响 attempt 才能继续计数。

### 只有 `.staging`，没有已发布 raw

说明当前物体还没通过全量审计与提交。重新运行会丢弃这部分临时状态，并用相同 seed 重跑该物体；
这是预期的安全行为。`.staging` 不能手工移动到正式 `raw/`。

### 有 partial `final_valid/`，没有 manifest

不要把 partial 目录当成完成数据。collector 在配额满足后会删除并确定性重建 `final_valid/`，然后
原子写入 `manifest.json`；原始 validated attempt 不受影响。

## 每次恢复后的强制审计

1. 顶层 collector 只有一个，GPU worker 数不超过 `--jobs 2`。
2. `attempt.json` 仍是 schema 4，raw target、seed、v6/v7 和 42-object catalog 没有变化。
3. raw 文件仍为 `success=false`、`validation.status=not_run`；不能把 raw 当 valid。
4. 每个已完成 attempt 都有可重新计算的 `complete.json`，且 raw = valid + failed。
5. 新 attempt 只由 collector 根据真实分层缺口与已观测 valid 率创建。
6. 最终 manifest 必须恰好 5000，front/back 各 2500，每侧 f1--f5 各 500。
7. front f1↔back f4、f2↔b3、f3↔b2、f4↔b1 的物体可以不同，但 finger set 必须互补且不相交；f5 单侧。
8. 最终选择必须覆盖固定 30 个通用 mesh，不能用 primitive 或重复少数物体代替。

## 运行日志追加格式

在每个启动、恢复、attempt 完成或故障后，向本节追加一行，并把详细实验结果同步到
[X2 抓取数据采集实验日志](x2_experiment_log.md)：

| 时间 | 事件 | attempt | raw/valid/failed | 当前对象或阶段 | 恢复动作 | 证据 |
|---|---|---|---|---|---|---|
| 2026-07-16 17:06 | 正式启动 | 0000 | 0/0/0 | 生成；sphere_r020/r030 | 新运行 | `attempt_0000/attempt.json` |
| 2026-07-16 17:19 | 健康检查 | 0000 | 0/0/0 | 两个生成 worker 正常；尚未到原子提交点 | 无 | 进程/GPU 快照 |
| 2026-07-16 17:24 | 恢复机制测试 | 0000 | 0/0/0 | 正式进程未中断；临时目录单测 | 6/6 通过 | 见“恢复机制验证” |
| 2026-07-16 18:01 | 账号交接预案 | 0000 | 0/0/0 | 正式进程未中断；两个 worker 正常 | 新增新账号接手提示 | `x2_codex_handoff.md` |
| 2026-07-16 18:08 | 首批原子提交与守护 | 0000 | 300/0/0 | sphere_r020/r030 完成；下一批生成 | 300 条审计 0 error；启用 systemd user supervisor | `collector_supervisor.log` |
| 2026-07-16 18:23 | 自动最终审计回归 | 0000 | 300/0/0 | 生成继续；manifest 尚不存在 | 15/15 collector+audit+supervisor 测试；服务已热重载 | 最终审计（不公开） |
| 2026-07-16 18:24 | 正式 catalog 独立复核 | 0000 | 300/0/0 | 12 primitive + 30 general | 42 个 mesh 文件/scale/SHA 全匹配；30 个 ID 唯一 | `attempt_0000/attempt.json` |
| 2026-07-16 22:50 | 12 primitive 完成 | 0000 | 1800/0/0 | 开始 general 000/003 | 每侧 f1--f5 各 180；1775 dense feasible、25 将静态失败 | raw 全量复核 |
| 2026-07-17 20:40 | PhysX OOM 根因确认 | 0000 | 6250/308/2242 | general `012` 完成 | batch 32/16 均 OOM；实测 batch 8 完成剩余 134 条 | `collector_console.log` + `validation_summaries/012.json` |
| 2026-07-17 20:56 | fail-fast 版本受控恢复 | 0000 | 6250/476/3464 | 从 partial `042` 恢复 | 固定 batch 8；不自动降批/限流；复用已落盘 route | systemd journal + `collector_console.log` |
| 2026-07-17 21:14 | PhysX 健康快照 | 0000 | 6250/624/4926 | 37/42 物体完成；正在 `075` | 5550/6250 已路由；无新 OOM 或参数回退 | 37 个 `validation_summaries/*.json` |
| 2026-07-17 21:19 | attempt 完成 | 0000 | 6250/642/5608 | 42/42 物体完成；10.27% valid | completion proof 通过；7 个 general mesh 为 0 valid | `attempt_0000/complete.json` |
| 2026-07-17 21:19 | 自适应补采启动 | 0001 | 0/0/0 | 生成；seed 1009 | 按真实缺口计划 66757 raw；无 cap | `attempt_0001/attempt.json` |
| 2026-07-17 21:23 | passed 样本 GUI 重放 | 0000 | 6250/642/5608 | general 030 / back f2 | identity passed；0.311 mm 位移；11.59 N 接触力 | Isaac Sim viewer stdout |
| 2026-07-17 21:31 | ETA 重算 | 0001 | target 66757 raw | 首批 sphere worker 运行 | 生成 73.5--261.8 h；PhysX 9--12 h；中心区间 6--9 天 | 首轮 24.51 h + batch-8 7656 raw/h |
| 2026-07-17 21:58 | 30 分钟长批健康检查 | 0001 | 0/0/0 | sphere_r020/r030 仍在内存优化 | 两 worker 持续 100% CPU、GPU 75--87%、无 OOM/退出；不重启 | 进程/GPU/文件快照 |
| 2026-07-17 22:04 | 独立终端仪表盘 | 0001 | 6250/642/5608 completed pool | 首批 sphere worker 继续 | 4 项相关测试通过；真实 `--once` 快照复核通过 | `scripts/watch_x2_collection.py` |
| 2026-07-17 22:22 | supervisor 健康日志修复 | 0001 | 0/0/0（本轮） | sphere_r020/r030 继续满载生成 | 自有 child 改为只读周期日志；11 项相关测试通过；不重启当前任务 | supervisor 日志（实现不公开） |
| 2026-07-26 14:21 | 登录后自动续采 | 0001 | 60421/0/0（本轮） | 380/420 group；生成 078/081 | supervisor 同参数恢复；复用全部完整 group | `collector_supervisor.log` |
| 2026-07-26 14:53 | 提前验证 fail-fast | 0001 | 60421/0/0（启动前） | Raw schema 预扫描 | 暴露 fallback/checkpoint 交叉契约错误；未启动仿真、未写正式 summary | `early_physx_60421/console.log` |
| 2026-07-26 15:03 | 提前 PhysX 启动 | 0001 | 60421；首批 1/191 valid/failed | sphere_r020；生成 078/081 并行 | 修复 112 条合法 fallback 的加载契约；30/30 测试；隔离 summary | `x2-early-physx-attempt1.service` |
| 2026-07-26 15:32 | 同-attempt验证互斥启用 | 0001 | sphere_r020 177/1417 | sphere_r030；生成 078/081 并行 | `.physx_validation.lock`；重启后 1594 条全量复用；正式 validator 将等待并续验 | early service + sphere_r020 summary |

追加示例：

```markdown
| YYYY-MM-DD HH:MM | 恢复/完成/故障 | NNNN | raw/valid/failed | 阶段或对象 | 执行的同参数恢复动作 | complete.json/summary/log 路径 |
```

## 实现依据与限制

- collector：[collect_x2_valid_dataset.py](../scripts/collect_x2_valid_dataset.py)
- 分层生成与 group resume：[generate_x2_primitive_dataset.py](../scripts/generate_x2_primitive_dataset.py)
- PhysX wrapper resume / 单物体 v7 原子路由 / 最终独立全量审计：实现不随公开仓库分发
- 当前 attempt metadata：
  [`attempt_0000/attempt.json`](../data/x2_valid_5000/attempts/attempt_0000/attempt.json)
- 更换 Codex 账号或新会话的接手提示：[Codex 账号交接](x2_codex_handoff.md)

本手册不能代替运行时证据。进程列表只证明“正在运行”，raw 数只证明“已生成”，valid 路由只
证明“单条已通过”；只有 completion proof 与最终 manifest 能证明正式进度和最终完成。

## 恢复机制验证

2026-07-16 17:24 CST 在独立临时目录运行 6 个恢复单元测试，6/6 通过、耗时 0.105 秒；没有
暂停或修改 `data/x2_valid_5000`。覆盖范围如下：

- 未完成 attempt 在创建新 attempt 之前优先恢复；
- stratified resume 可复用 39/40 个完整 group，仅重建 1 个缺失 group；
- 完整 group 保持原字节，缺失/损坏 group 重建，并删除由旧 raw 产生的 stale route；
- 并发生成中一个任务失败时，已经成功提交的 group 保留，未启动 future 被取消；
- PhysX wrapper 恢复时发布全量 scanned valid/failed 计数；
- 旧协议 route 被识别、删除并按当前 v7 重新验证。

对应单元测试未随公开仓库分发，2026-07-16 在独立临时目录执行结果为 6/6 通过。

这组测试证明代码级 resume 契约，但不假装等同于真实断电演练；真实演练会等至少一个正式物体
完成提交后再决定，避免无收益地丢弃当前两个内存 batch。

## 下一步与待回答问题

1. 继续 `attempt_0001` 剩余 6336 raw 生成，同时观察提前 PhysX 的原子 route。
2. 生成完成后继续固定 batch 8 的正式 PhysX `--resume`，复核已有 route 并补验剩余对象。
3. 每个 completed attempt 后重算 f1--f5 缺口、互补 pair 和 30-mesh 覆盖。
4. 保留 passed 记录的 Isaac Sim GUI 重放流程，不用 raw 图片充当 valid 证据。
