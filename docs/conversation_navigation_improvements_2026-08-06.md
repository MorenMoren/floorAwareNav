# FloorAwareNav 当前对话改进总结

更新日期：2026-08-06

## 1. 文档范围

本文档总结本轮对 ASCENT 多楼层导航、OpenFrontier/FrontierNet 语义楼梯策略、普通探索 frontier 防卡死和 benchmark 稳定性所做的改进，并明确当前仍存在的限制。

改造遵循“最少干预原导航框架”的原则：

- 保留 ASCENT 的楼梯检测、楼层地图、跨楼层状态机和 PointNav 动作模型。
- 只在 `climb_stair` 阶段替换或增强短程子目标生成与目标生命周期。
- OpenFrontier、FrontierNet 或 VLM 不可用时回退到原始胡萝卜点。
- 普通目标物体探索 frontier 与语义楼梯 frontier 分开管理。

## 2. 当前导航链路

```text
普通探索
  └─ 常规地图 frontier → VLM/规则选点 → PointNav
                         └─ 空间进度守卫判断不可达目标

检测到楼梯
  └─ 导航到楼梯 centroid
       └─ 到达 centroid
            ├─ FrontierNet 有候选
            │    └─ Qwen 上/下楼语义评分
            │         └─ 锁定单一世界坐标子目标 → PointNav
            └─ 无有效候选
                 └─ legacy carrot → 高度方向校验 → PointNav/旋转搜索

楼梯区域判断不稳定
  └─ 3D GPS 高度窗口确认仍在爬升/下降 → 延迟退出 climb_stair
```

## 3. 已完成工作

### 3.1 OpenFrontier/FrontierNet 语义楼梯子目标

新增 `ascent/stair_policy/` 模块，将 OpenFrontier 浓缩为只负责楼梯通过的局部策略：

1. 进入 `climb_stair` 并到达楼梯 centroid。
2. FrontierNet 从当前 RGB-D 视图检测视觉 frontier。
3. 深度和相机位姿将候选锚定为 episodic 世界坐标。
4. 候选标记为 A、B、C，并生成 Set-of-Marks 图像。
5. 复用 ASCENT 已有的多模态 `Qwen/Qwen3.6-27B`，将原“寻找目标物体”语义替换为：
   - 上楼时选择继续向上的可通行台阶或平台方向；
   - 下楼时选择继续向下的可通行台阶或平台方向；
   - 排除墙、栏杆、侧向走廊、反方向楼梯和不可通行区域。
6. 根据语义概率、信息增益、距离、相对朝向和楼梯 mask 支持度过滤排序。
7. 只把一个最优候选交给原 PointNav。

主要文件：

- `ascent/stair_policy/openfrontier_adapter.py`
- `ascent/stair_policy/semantic_frontier_policy.py`
- `ascent/stair_policy/stair_vlm.py`
- `ascent/ascent_policy.py`

### 3.2 坐标系修正

OpenFrontier/OpenCV 相机坐标为：

```text
[x, y, z] = [右, 下, 前]
```

ASCENT 局部相机坐标为：

```text
[x, y, z] = [前, 左, 上]
```

已加入显式坐标基转换：

```text
[x_ascent, y_ascent, z_ascent] = [z_cv, -x_cv, -y_cv]
```

该修正解决了 FrontierNet 候选在 obstacle/top-down map 上明显偏移，以及 PointNav 因目标横向错位而持续右转的问题。

### 3.3 候选过滤和日志

当前默认硬过滤包括：

- 语义概率不低于 `0.15`。
- 距离位于 `[0.3m, 2.5m]`。
- 相对朝向不超过约 `125°`。
- 世界坐标有效。
- 不靠近已经标记为 blocked 的语义楼梯目标。

每次新检测会打印所有 A/B/C 候选，而不只打印最终选中的一个：

```text
Semantic stair candidates:
A(p=0.90, valid, goal=[...], filter=valid) |
B(p=0.00, rejected, goal=[...], filter=semantic probability below threshold)
```

随后单独打印最终选中的候选、utility、世界坐标和 VLM 原因。

### 3.4 语义目标锁定和检测频率

选中 FrontierNet 目标后，该点会作为世界坐标短程 PointNav 目标保持锁定：

- 距离目标大于 `0.8m` 时不重新运行 FrontierNet/VLM。
- 目标不会随着机器人位姿或当前帧重新投影而漂移。
- 到达 `0.8m` 内后释放目标，下一段才允许重新检测。
- 空检测和全部拒绝结果也记录检测时间，默认至少间隔 4 step 才重试，避免每步调用 Qwen。
- 无有效候选时立即回退 legacy carrot，不继续追逐旧语义目标。

这使 FrontierNet 点真正成为“分段锁定的临时导航目标”，而不是每帧变化的胡萝卜点。

### 3.5 视频与地图可视化

FrontierNet 候选已加入 debug 视频：

- RGB 画面显示 A/B/C Set-of-Marks。
- Habitat top-down map 使用候选的 episodic 3D 锚点进行投影。
- 青色表示有效未选候选，橙色表示被过滤候选，紫色实心点表示最终选中候选。
- 图像预处理采用 resize + center crop，显示时执行逆变换，减少候选在原始 RGB 上的位置偏差。
- 地图绘制处理 episodic-to-global、Habitat 坐标转换、地图旋转和最终缩放。

可视化字段只用于 debug，控制策略不会反向读取这些绘制结果。

### 3.6 上楼相机生命周期

上楼语义目标采用以下相机循环：

1. 没有 active goal 时先回到平视 `pitch=0`。
2. 平视状态运行 FrontierNet。
3. 选中语义目标后执行 `LOOK_UP`，再交给 PointNav。
4. 到达目标 `0.8m` 内后释放目标。
5. 执行 `LOOK_DOWN` 回到平视，再检测下一段。

这样减少了 PointNav 因楼梯画面占比过高而长期转向、不直接导航到锁定目标的问题。

### 3.7 基于高度的楼梯状态保持

HM3D 和 MP3D eval 配置已将 episodic GPS 设置为三维。高度只用于楼梯状态判断，不加入 ASCENT 原有的每层局部地图投影，以避免上层点云被固定高度过滤器丢弃。

当二维 obstacle map 暂时认为机器人离开楼梯区域时：

- 上楼且高度仍上升：保持 `climb_stair`。
- 下楼且高度仍下降：保持 `climb_stair`。
- 只有高度稳定且地图外状态连续确认后，才允许结束爬楼。
- 如果环境仍提供二维 GPS，则自动回退原地图判断。

该机制缓解了楼梯 mask 很小、机器人刚离开旧 mask 就提前切出爬楼状态的问题。

主要实现：`ascent/stair_policy/height_guard.py` 和 `ascent/map_controller.py`。

### 3.8 legacy carrot 的高度方向校验

当 FrontierNet 无有效候选、系统回退原始胡萝卜点时，增加了独立的高度方向守卫：

- 默认允许 PointNav 执行最多 4 次前进探测。
- 上楼要求高度至少增加 `0.03m`。
- 下楼要求高度至少降低 `0.03m`。
- 高度向相反方向变化时立即拒绝当前胡萝卜。
- 无正确方向高度变化时，清空胡萝卜并连续右转 3 步。
- 转动完成后基于新视野生成新胡萝卜并再次探测。
- 如果仍不正确，则重复“旋转—重新生成—前进探测”。
- 主动旋转期间清零 legacy 爬楼卡顿计数，避免方向搜索被误判为爬楼失败。
- 已由 VLM 验证的语义 FrontierNet 目标不受该守卫影响。

### 3.9 普通探索不可达 frontier 防卡死

普通 `explore` frontier 增加了空间进度守卫，解决精确浮点坐标黑名单容易被漂移点绕过的问题：

- 将 `0.6m` 内的 frontier 视为同一空间目标区域。
- 使用机器人到固定区域锚点的历史最小距离判断真实进展。
- 至少改善 `0.15m` 才重置无进展计数。
- 连续 20 次选择没有取得新进展后，将该区域判为不可达。
- 屏蔽锚点周围 `0.75m` 内后续重新生成的 frontier。
- 清空强制 frontier、PointNav 隐状态和当前目标，然后重新规划。
- 不可达区域按楼层隔离，避免相同 XY 坐标误伤其他楼层。
- 即使当前只剩一个 frontier，也会经过该守卫。

主要实现：`ascent/frontier_progress_guard.py`。

### 3.10 Benchmark 指标汇总崩溃修复

修复了 episode 结束时的错误：

```text
TypeError: float() argument must be a string or a number, not 'NoneType'
```

原因是 `policy_info` 合并进 `infos[i]` 后包含嵌套 `None`，Habitat 的通用指标提取器会执行 `float(None)`。当前 Ascent 适配层会递归过滤 `None` 和列表，再交给 Habitat 原提取器，并首次打印被过滤字段路径。

后续 `VectorEnv.__del__` 异常只是主异常触发后的子进程清理连带错误。

主要实现：`ascent/ascent_trainer.py`。

### 3.11 依赖、源码和权重

已补充：

- `third_party/OpenFrontier` 源码。
- FrontierNet 权重：`third_party/OpenFrontier/model_weights/rgbd_11cls.pth`。
- 可选依赖清单：`requirements_stair_frontier.txt`。
- 复用现有 Qwen OpenAI-compatible client，没有启动第二个 VLM。
- 安装过程中保持 PyTorch、NumPy 和 Habitat-Sim 版本不变。

## 4. Benchmark 失败模式和处理状态

| 编号 | 失败模式 | 当前状态 |
|---|---|---|
| BENCH-001 | 普通 frontier 实际不可达，机器人持续尝试 | 已增加空间聚类进度守卫和邻域黑名单；尚未加入 A*/FMM 静态可达性检查 |
| BENCH-002 | 已到目标附近，但检测模型漏检，任务不能停止 | 已记录，尚未完成多帧目标确认或 VLM 停止恢复 |
| BENCH-003 | 楼梯识别漏检，无法触发跨楼层导航 | 已记录；语义爬楼策略仍依赖上游楼梯检测触发 |
| BENCH-004 | 楼梯 mask 太小，离开旧区域后提前退出爬楼 | 已用三维 GPS 高度窗口进行缓解 |
| BENCH-005 | legacy carrot 指向错误上/下楼方向 | 已增加高度方向探测和旋转搜索 |
| BENCH-006 | FrontierNet 点随机器人位置变化 | 已改为 episodic 世界坐标锁定，并修复 OpenCV/ASCENT 坐标基 |
| BENCH-007 | 评测信息含 `None` 导致 episode 汇总崩溃 | 已递归过滤非标量调试字段 |

## 5. 当前已知限制

### 5.1 下楼相机逻辑尚未与上楼完全对称

当前下楼流程是：

- 接近 centroid 时先低头，通常从 `0°` 到 `-30°`，可能继续到 `-60°`。
- 到达 centroid 后如果低于 `-30°`，会执行一次 `LOOK_UP` 回到约 `-30°`。
- FrontierNet 选中下楼目标后没有额外 `LOOK_DOWN`。
- 语义策略仍会在到达 `0.8m` 后释放目标，但下楼不会像上楼一样先恢复到严格平视再检测下一段。

因此下楼当前通常在约 `-30°` 检测和导航，而不是“平视检测 → 选点后低头 → 到达后恢复平视”的对称循环。这是下一步应优先补齐的状态机逻辑。

### 5.2 高度只能验证运动结果

机器人高度不能在移动前直接证明胡萝卜方向正确，因此 legacy carrot 使用“短程前进探测 + 旋转重试”。完全禁止任何试探前进会使高度信号无法产生。

### 5.3 不可达 frontier 当前是动态判定

普通 frontier 守卫需要观察若干次无进展后才会禁用目标。导航前使用可通行连通域、A* 或 FMM 的静态可达性检查尚未实现。

### 5.4 感知仍是跨楼层触发瓶颈

FrontierNet 改进的是“已经进入爬楼状态后往哪里走”，不能替代楼梯检测器本身。若 RedNet、目标检测或楼梯 mask 没有产生有效楼梯区域，日志仍会保持：

```text
Mode: explore | Stair_flag: 0
```

此时语义楼梯策略、相机循环和高度胡萝卜守卫均不会运行。

## 6. 主要参数

```bash
# 语义楼梯 FrontierNet/VLM
export ASCENT_SEMANTIC_STAIR_ENABLED=1
export ASCENT_STAIR_REPLAN_INTERVAL=4
export ASCENT_STAIR_GOAL_REACHED_DISTANCE=0.8
export ASCENT_STAIR_MIN_SUBGOAL_DISTANCE=0.3
export ASCENT_STAIR_MAX_SUBGOAL_DISTANCE=2.5
export ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY=0.15
export ASCENT_STAIR_MAX_HEADING_DEG=125

# 楼梯退出高度保护
export ASCENT_STAIR_HEIGHT_WINDOW_STEPS=5
export ASCENT_STAIR_HEIGHT_CHANGE_THRESHOLD=0.03
export ASCENT_STAIR_EXIT_STABLE_STEPS=6

# legacy carrot 方向验证
export ASCENT_CARROT_HEIGHT_PROBE_FORWARD_STEPS=4
export ASCENT_CARROT_HEIGHT_CHANGE_THRESHOLD=0.03
export ASCENT_CARROT_DIRECTION_ROTATE_STEPS=3

# 普通探索 frontier 不可达检测
export ASCENT_FRONTIER_ASSOCIATION_RADIUS=0.6
export ASCENT_FRONTIER_BLOCKED_RADIUS=0.75
export ASCENT_FRONTIER_MIN_PROGRESS=0.15
export ASCENT_FRONTIER_MAX_STALLED_SELECTIONS=20

# 现有多模态 Qwen 服务
export ASCENT_QWEN_MODEL=Qwen/Qwen3.6-27B
export ASCENT_QWEN_BASE_URL=http://your-qwen-server/v1
export ASCENT_QWEN_API_KEY=EMPTY
```

## 7. 关键日志判读

### 普通探索，不会触发楼梯策略

```text
Mode: explore | Stair_flag: 0
```

### 已进入语义爬楼

```text
Mode: climb_stair | Stair_flag: 1
Semantic stair candidates: A(...) | B(...)
Semantic stair frontier A: ...
```

### 语义目标保持锁定

```text
Semantic Stair Goal - Distance: ... Progress: ... Locked: True
```

### legacy carrot 方向错误

```text
Legacy stair carrot rejected by height: ... turning right.
```

### 普通 frontier 判为不可达

```text
Exploration frontier marked unreachable: ...
Blocking nearby regenerated frontiers and replanning.
```

## 8. 验证情况

当前单元测试覆盖：

- OpenFrontier 适配和坐标转换。
- FrontierNet 候选缓存、过滤、VLM JSON 解析和目标锁定。
- RGB/top-down map 候选可视化。
- 楼梯高度退出保护。
- legacy carrot 上楼/下楼方向校验和旋转重试。
- 普通 frontier 漂移聚类、无进展检测、邻域屏蔽和楼层隔离。
- episode info 嵌套 `None` 过滤。

截至 2026-08-06，完整测试结果：

```text
Ran 31 tests
OK
```

测试命令：

```bash
PYTHONPATH=.:third_party/vlfm:third_party/frontier_exploration:third_party/depth_camera_filtering \
  /data/miniconda3/envs/ascent_nav/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

## 9. 版本状态

已推送到 `MorenMoren/floorAwareNav` 的基线提交：

```text
75a50bf feat: improve semantic stair navigation robustness
```

该提交之后新增的 legacy carrot 高度方向校验、普通探索不可达 frontier 守卫及本文档目前仍位于本地工作区，尚未包含在上述远端提交中。

## 10. 推荐下一步

1. 将下楼相机流程改为与上楼对称的“平视检测—选点后低头—到达后恢复平视”。
2. 在普通 frontier 进入 PointNav 前增加 navigable connected component 或 A*/FMM 静态可达性验证。
3. 为目标物体漏检增加多帧证据、近距离主动观察和 VLM 恢复终止机制。
4. 对楼梯检测结果做跨帧融合，避免单帧小 mask 或漏检阻断跨楼层触发。
5. 在 benchmark 输出中增加失败原因计数：不可达 frontier、目标漏检、楼梯漏检、楼梯提前退出和方向搜索次数。
