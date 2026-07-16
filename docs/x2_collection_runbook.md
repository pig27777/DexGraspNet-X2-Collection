# X2 正式数据采集运行日志与断点恢复手册

最后更新：2026-07-16 22:50 CST

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
| 启动时间 | 2026-07-16 17:06 CST |
| 输出根目录 | `data/x2_valid_5000` |
| 当前 attempt | `attempt_0000` |
| raw 计划 | 6250；f1、f2、f3、f4、f5 各 1250 |
| catalog | 12 primitive + 固定 30 general mesh |
| 生成协议 | `x2_mesh_grasp_unselected_finger_side_v6`，6000 iterations |
| 验证协议 | `x2_object_centered_dexgraspnet_six_orientation_v7` |
| 采集协议 | `x2_balanced_complementary_30mesh_5000_v6` |
| 22:50 状态 | 12 个 primitive 已原子发布；正在生成首批通用 mesh `000`、`003` |
| 已发布 raw/valid/failed | 1800/0/0；首个 attempt 的 raw 生成进度 28.8% |
| raw 静态审计 | 1800 finite、自碰撞通过；1775 dense `<1 mm`，25 dense infeasible |
| raw 分层/配对 | front/back × f1--f5 每格 180；720 个生成互补 pair，其中 696 对双方 dense feasible |
| 正式计数 | 0/5000；`attempt_0000` 尚无 `complete.json` |

当前没有趋势图：还没有 completed attempt，partial raw 不能作为正式吞吐或有效率分母。每个
attempt 完成后再追加同协议下的 raw/valid/failed 与 side/finger_count 表格。

## 跨会话后台守护

守护器代码为 `scripts/supervise_x2_valid_collection.py`，持久用户单元为
`systemd/x2-valid-collector-supervisor.service`。该单元已链接并启用，Codex 账号或终端切换不会
依赖旧对话的 PTY；若操作系统重启，它会在该 Linux 用户下次登录时恢复采集。

```bash
systemctl --user status x2-valid-collector-supervisor.service --no-pager
tail -f data/x2_valid_5000/collector_supervisor.log
```

守护器每 15 秒探测 `.collector.lock`，每 5 分钟把已发布 raw/valid/failed 写入持久日志。它只在
锁空闲且 manifest 没有严格证明 5000 条配额时恢复；manifest 的 headline 配额满足后会自动
运行 `audit_x2_valid_dataset.py`。只有全量审计报告与当前 manifest SHA-256 绑定且通过，服务才
发布 `final_audit.json` 并退出；失败报告保存为 `final_audit_failed.json`，不能提前结束 goal。

若要**主动暂停**，必须先阻止守护器重启任务：

```bash
touch data/x2_valid_5000/.stop_supervisor
systemctl --user stop x2-valid-collector-supervisor.service
pgrep -af 'collect_x2_valid_dataset.py|generate_x2_mesh_grasps_stratified.py|validate_x2_mesh_grasps_physx.py'
```

若 `pgrep` 仍显示本次 17:06 从旧 PTY 启动的 collector，再对其顶层 PID 发送一次 `SIGINT` 并等待
全部 child 退出。重新续采时删除 sentinel 并启动服务：

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

pgrep -af 'collect_x2_valid_dataset.py|generate_x2_primitive_dataset.py|generate_x2_mesh_grasps_stratified.py|validate_x2_primitive_dataset.py|validate_x2_mesh_grasps_physx.py'

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
  --validation-batch-size 32 \
  --sim-steps 100 \
  --general-mesh-root data/meshdata \
  --output-root data/x2_valid_5000 \
  2>&1 | tee -a data/x2_valid_5000/collector_console.log
```

`tee -a` 让后续恢复运行拥有持久控制台日志；`set -o pipefail` 保证 collector 失败时整条管道也
返回失败。当前 17:06 启动的首个进程没有经过 `tee`，不应为了补日志而主动停止；它的权威状态
仍由 attempt 文件证明。

正常恢复开头应出现类似：

```text
[collector] resuming attempt_0000
[resume] reusable_groups=<已提交层数> regenerate_groups=<待生成层数>
```

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
30 个通用 mesh 全覆盖。

manifest 出现后必须运行独立只读审计器；退出码为 0 且报告 `passed=true` 才能结束 goal：

```bash
set -o pipefail
conda run -n isaaclab --no-capture-output \
  python scripts/audit_x2_valid_dataset.py \
  --output-root data/x2_valid_5000 \
  --general-mesh-root data/meshdata \
  | tee data/x2_valid_5000/final_audit.json
```

该审计器不信任 manifest 的 headline：它重新哈希全部 5000 个 final 文件、验证 hard link 与
source、一一重跑 v6/v7 JSON 契约检查、重算 attempt completion proofs、连续分层索引、2000 个
同物体互补 pair、1000 个 f5 单侧条目以及固定 30-mesh 覆盖。当前 manifest 尚不存在时，它按
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

先确认 GPU 上没有残留生成/验证进程，再用 `nvidia-smi` 检查显存。不要直接降低 batch、修改
drive 或放宽 v7 门槛来接着写同一 attempt；参数改变会破坏可比性。先保存错误日志，再按同参数
重试；相同对象连续复现时才进入独立故障实验和代码修复。

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
7. front f1↔back f4、f2↔b3、f3↔b2、f4↔b1 必须同物体且 finger set 不相交；f5 单侧。
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
| 2026-07-16 18:23 | 自动最终审计回归 | 0000 | 300/0/0 | 生成继续；manifest 尚不存在 | 15/15 collector+audit+supervisor tests；服务已热重载 | `audit_x2_valid_dataset.py` |
| 2026-07-16 18:24 | 正式 catalog 独立复核 | 0000 | 300/0/0 | 12 primitive + 30 general | 42 个 mesh 文件/scale/SHA 全匹配；30 个 ID 唯一 | `attempt_0000/attempt.json` |
| 2026-07-16 22:50 | 12 primitive 完成 | 0000 | 1800/0/0 | 开始 general 000/003 | 每侧 f1--f5 各 180；1775 dense feasible、25 将静态失败 | raw 全量复核 |

追加示例：

```markdown
| YYYY-MM-DD HH:MM | 恢复/完成/故障 | NNNN | raw/valid/failed | 阶段或对象 | 执行的同参数恢复动作 | complete.json/summary/log 路径 |
```

## 实现依据与限制

- collector：[collect_x2_valid_dataset.py](../scripts/collect_x2_valid_dataset.py)
- 分层生成与 group resume：[generate_x2_primitive_dataset.py](../scripts/generate_x2_primitive_dataset.py)
- PhysX wrapper resume：[validate_x2_primitive_dataset.py](../scripts/validate_x2_primitive_dataset.py)
- 单物体 v7 原子路由：[validate_x2_mesh_grasps_physx.py](../scripts/validate_x2_mesh_grasps_physx.py)
- 最终独立全量审计：[audit_x2_valid_dataset.py](../scripts/audit_x2_valid_dataset.py)
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

执行命令：

```bash
conda run -n isaaclab --no-capture-output python -m unittest -v \
  tests.test_x2_valid_collector.X2ValidCollectorTest.test_incomplete_attempt_is_resumed_before_new_work \
  tests.test_x2_primitive_dataset.X2PrimitiveDatasetTest.test_stratified_resume_batches_per_object_and_repairs_one_group \
  tests.test_x2_primitive_dataset.X2PrimitiveDatasetTest.test_resume_reuses_complete_groups_and_repairs_only_damaged_groups \
  tests.test_x2_primitive_dataset.X2PrimitiveDatasetTest.test_resume_failure_checkpoints_success_and_cancels_unstarted_futures \
  tests.test_x2_isaac_validation.X2IsaacValidationTests.test_primitive_wrapper_resume_publishes_full_scanned_counts \
  tests.test_x2_isaac_validation.X2IsaacValidationTests.test_primitive_wrapper_resume_revalidates_stale_v5_route
```

这组测试证明代码级 resume 契约，但不假装等同于真实断电演练；真实演练会等至少一个正式物体
完成提交后再决定，避免无收益地丢弃当前两个内存 batch。

## 下一步与待回答问题

1. 首个物体原子提交后记录实际每物体生成耗时，并校准 `attempt_0000` ETA。
2. `attempt_0000` 完成后记录 f1--f5/front-back 的有效率和主要失败原因。
3. 验证一次真实中断恢复演练应在何时进行：优先等待至少一个物体已提交，避免为了演练主动
   丢弃当前两个仍在内存中的长批次。
4. 当前环境没有安装 `tmux`；本轮不改变正在运行的进程。后续若需要跨终端持久会话，可单独
   增加受控服务或会话管理，但它不能取代 collector 自身的 resume 与 completion 审计。
