# FloorAwareNav：基于 OpenFrontier/FrontierNet 的语义爬楼策略与 Benchmark 失败分析

更新日期：2026-08-04

## 1. 文档目的

本文档总结本轮开发中对 ASCENT 多楼层导航和爬楼梯策略的改进，并统一记录 benchmark 评测过程中观察到的主要失败模式、代码层原因及可能的解决方向。

需要区分两类 frontier：

- **常规探索 frontier**：由 ASCENT 障碍物地图上的已探索/未探索边界生成，用于目标物体探索。
- **语义楼梯 frontier**：仅在 `climb_stair` 阶段由 FrontierNet 从当前 RGB-D 视图生成，用于替换原来的短程胡萝卜点。

当前实现只改造第二类 frontier，不改变 ASCENT 原有目标探索、PointNav 动作模型和楼层地图切换框架。

## 2. 原始 ASCENT 爬楼梯机制

原始跨楼层流程大致为：

1. 目标检测器、SAM 和 RedNet 识别楼梯区域。
2. 将图像楼梯区域投影到 `_up_stair_map` 或 `_down_stair_map`。
3. 使用楼梯区域 centroid 作为接近楼梯的目标。
4. 到达 centroid 后，根据深度图中最远点生成固定距离约 `0.8m` 的胡萝卜点。
5. 将胡萝卜点交给原有 PointNav 策略产生离散动作。
6. 根据机器人是否离开楼梯地图、是否卡住等状态决定楼梯结束和楼层切换。

该机制的优点是控制链简单，但胡萝卜点仅利用局部深度，没有判断候选方向是否真的沿楼梯向上或向下，容易把墙面、平台侧边或错误通道作为下一目标。

## 3. 改进目标与最小干预原则

本次改进参考 OpenFrontier 的思想，将“寻找目标物体的语义 frontier”改造为“寻找上楼/下楼可通行方向的语义 frontier”。

核心原则是只替换短程子目标生成器：

```text
楼梯检测与模式触发                         保持 ASCENT 原逻辑
            ↓
FrontierNet + Qwen 生成语义楼梯子目标       本次新增
            ↓
PointNav 将子目标转换为离散动作             保持 ASCENT 原逻辑
            ↓
爬楼结束判定和楼层地图切换                  保持 ASCENT 原逻辑
```

这样可以在不重新训练 PointNav、不重写多楼层状态机的前提下替换原胡萝卜点策略，并在新模块不可用时安全回退。

## 4. 语义楼梯 Frontier 策略

### 4.1 完整运行链路

进入 `climb_stair` 且机器人已经到达楼梯 centroid 后：

1. 读取当前 RGB、深度、相机内参和相机位姿。
2. FrontierNet 检测当前视图中的潜在视觉 frontier。
3. 使用深度将每个 2D frontier 锚定为 episodic 坐标系中的 3D 点。
4. 为候选分配当前帧临时标签 A、B、C，并生成 Set-of-Marks 图像。
5. 复用 ASCENT 已有的多模态 `Qwen/Qwen3.6-27B`，对每个候选判断其是否：
   - 沿当前楼梯继续向上或向下；
   - 位于可通行的台阶或平台表面；
   - 能够朝目标楼层产生有效进展；
   - 不是墙面、栏杆、障碍物、侧向走廊或反方向楼梯。
6. 使用语义概率、距离、相对朝向、信息增益和楼梯区域支持度过滤并排序候选。
7. 只选择一个最优短程 XY 子目标交给原有 PointNav。
8. 如果无有效候选、FrontierNet/VLM 不可用或推理失败，则使用原始胡萝卜策略。

### 4.2 坐标系修正

FrontierNet/OpenCV 的相机点坐标为：

```text
[x, y, z] = [右, 下, 前]
```

ASCENT/VLFM 使用的局部相机坐标为：

```text
[x, y, z] = [前, 左, 上]
```

因此在应用 ASCENT 相机位姿前必须完成：

```text
[x_ascent, y_ascent, z_ascent] = [z_cv, -x_cv, -y_cv]
```

早期版本直接反转 ASCENT 相机位姿并传给 OpenFrontier，导致 frontier 在地图上横向偏移，PointNav 认为目标持续位于机器人右侧，从而连续输出 `Action: 3`（右转）。当前适配器已经加入显式坐标基变换。

### 4.3 候选过滤与效用函数

当前主要硬过滤条件为：

- VLM 已评分候选的语义概率不低于 `0.15`。
- 子目标距离位于 `[0.3m, 2.5m]`。
- 候选不靠近已经标记为 blocked 的目标。
- 候选相对机器人朝向不超过约 `125°`。
- 候选 3D 坐标必须有效。

有效候选的效用可概括为：

```text
utility = p^α × gain^β × stair_support / distance^γ
          - turn_penalty × |relative_heading| / π
```

其中 `p` 是 Qwen 给出的上楼/下楼语义概率。设置语义概率下限是必要的，否则即使所有候选都是 `p=0.00`，取最大 utility 时仍会错误选择其中一个。

候选 A/B 标签只在当前检测帧内有效。下一次 FrontierNet 推理会重新编号，因此不同 step 的 A 不代表同一个空间目标。

### 4.4 锁定、分段检测和回退

默认配置：

```bash
ASCENT_STAIR_GOAL_REACHED_DISTANCE=0.8
```

当存在有效 active goal 时，目标在世界坐标系中保持锁定：

```text
Step N：FrontierNet + Qwen
选中目标：缓存世界坐标并执行一次 LOOK_UP
距离目标 > 0.8m：抬头状态下持续复用 active goal，不调用 FrontierNet/Qwen
距离目标 <= 0.8m：释放目标并执行 LOOK_DOWN 回到平视
恢复 pitch=0：检测下一个楼梯 frontier
```

这使 PointNav 有足够的步数到达开放楼梯区域中的目标，不会因为原先的
4-step 周期而用新相机视角反复改变目标。锁定期间即使短时没有距离进展，
也不会释放目标或回退到相对机器人生成的胡萝卜点。

锁定期间，旧的 centroid `climb_stair_paused_step` 保持为 0。策略改为记录
机器人到当前 semantic frontier 的绝对距离和有符号进展，避免旧的 30-step
centroid 超时在 PointNav 到达目标前终止爬楼，也避免机器人正常离开楼梯
区域时被误判为卡顿失败。

上楼采用“平视检测、抬头导航”的分段循环：到达 centroid 后先确保
`pitch=0`，仅在平视状态调用 FrontierNet/Qwen；选中候选并缓存坐标后执行
一次 `LOOK_UP`，随后由 PointNav 在抬头视野下导航到该固定点。进入目标
`0.8m` 范围后释放目标，执行 `LOOK_DOWN` 恢复平视，再检测下一段。
已经位于 `0.8m` 到达范围内的候选会被过滤，避免产生无需导航的抬头/低头
循环。下楼视角逻辑保持不变。

如果完成一个分段后的新检测没有有效候选，则当前 step 使用原始胡萝卜
策略。空候选或全部被拒绝的检测也会更新时间戳，默认每 4 step 重试，
不会再每一步调用 FrontierNet，也不会因为重复空检测将 legacy paused
counter 重置为 0。只有成功选取 semantic frontier 后才进入锁定状态。

为防止目标频繁切换，策略带有 hysteresis。负 utility 不能直接乘比例判断，新实现使用必须产生实际提升的加性 margin。

### 4.5 可视化与日志

debug 视频中：

- 青色：通过过滤的候选。
- 橙色：被语义或几何条件拒绝的候选。
- 实心品红：当前被选中的语义子目标。

候选同时显示在：

- FrontierNet Set-of-Marks/VLM 输入图；
- 主 RGB 画面；
- Habitat top-down map。

控制台先打印所有候选及过滤原因，再打印最终交给 PointNav 的唯一目标，例如：

```text
Semantic stair candidates:
A(p=0.90, valid, ...)
| B(p=0.00, rejected, filter=semantic probability below threshold)

Semantic stair frontier A: p=0.90, utility=..., goal=...
```

检测到 A、B 但只输出一个是预期行为：FrontierNet 负责产生候选，Qwen 和效用函数负责选择，PointNav 每次只接收一个子目标。

## 5. 代码结构

```text
ascent/stair_policy/
├── openfrontier_adapter.py       # OpenFrontier 加载、FrontierNet 推理、坐标转换
├── semantic_frontier_policy.py   # 候选过滤、效用、缓存、回退和 debug 状态
├── stair_vlm.py                  # 上楼/下楼语义提示词及 Qwen 结果解析
└── __init__.py

ascent/ascent_policy.py           # 在 climb_stair 中接入短程语义子目标
ascent/habitat_visualizer.py      # RGB、Set-of-Marks 和地图候选可视化
model_api/qwen36_out.py           # 与探索规划器共享的多模态 Qwen 客户端
third_party/OpenFrontier          # 官方 OpenFrontier submodule
tests/                            # 策略选择、坐标转换和可视化测试
```

FrontierNet 权重默认路径：

```text
third_party/OpenFrontier/model_weights/rgbd_11cls.pth
```

主要配置项：

```bash
export ASCENT_SEMANTIC_STAIR_ENABLED=1
export ASCENT_STAIR_GOAL_REACHED_DISTANCE=0.8
export ASCENT_STAIR_MIN_SUBGOAL_DISTANCE=0.3
export ASCENT_STAIR_MAX_SUBGOAL_DISTANCE=2.5
export ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY=0.15
export ASCENT_STAIR_MAX_HEADING_DEG=125
export ASCENT_STAIR_HYSTERESIS_RATIO=1.2
```

`ASCENT_STAIR_REPLAN_INTERVAL` 和 `ASCENT_STAIR_STUCK_STEPS` 为兼容旧配置而
保留，但不会导致已锁定的楼梯 semantic frontier 被提前释放或替换。

## 6. Benchmark 中观察到的失败模式

以下问题是评测过程中已经观察到或由日志确认的失败模式。除特别说明外，本节的解决方向尚未全部实现。

### BENCH-001：不可达的常规探索 Frontier 被持续选择

#### 现象

机器人把障碍物地图上的某个探索 frontier 作为导航目标，但该点实际上不可达。机器人持续转向、前进或重新规划到同一区域，最终耗尽 episode 步数。

#### 当前机制缺口

- 当只剩一个常规 frontier 时，LLM planner 直接返回，绕过卡住检测和禁用逻辑。
- `_disabled_frontiers` 使用精确浮点坐标；frontier 轻微漂移后可能绕过黑名单。
- 距离进展使用绝对差，机器人远离目标也可能被认为发生了移动。
- PointNav 对探索目标输出停止时会被强制改为前进，可能持续撞向不可达区域。

#### 可能解决方法

1. 所有候选（包括唯一 frontier）都必须经过统一目标生命周期管理。
2. 在导航前使用 `_navigable_map` 连通域、A* 或 FMM 检查可达性。
3. 将 frontier 吸附到机器人所在可通行连通域中的最近安全像素。
4. 使用有方向的进展 `previous_distance - current_distance`，在滑动窗口内判断是否真正接近目标。
5. 使用 `0.3～0.5m` 的空间黑名单，而不是禁用单个精确坐标。
6. 达到卡住阈值后执行 `blocked → 清除目标 → 重新选择`，不再盲目强制前进。

### BENCH-002：已到达目标物体但目标漏检，任务无法终止

#### 现象

机器人已经进入目标物体附近并获得包含目标的视角，但目标检测器没有产生目标框，策略继续探索并离开目标区域，benchmark 最终判定失败。

#### 当前机制缺口

- BLIP 图像相似度确认位于 `for target_detections` 循环内部；如果检测框数量为零，BLIP/VLM 无法独立触发目标确认。
- 目标确认依赖单帧检测，缺少主动观察和多帧证据融合。
- SAM mask 腐蚀或无效深度可能产生空点云。空点云被传给 Open3D DBSCAN 时会出现：

```text
[Open3D WARNING] [KDTreeFlann::SetRawData] Failed due to no data.
```

此警告通常不会导致程序崩溃，但当前目标观测会被忽略，从而进一步增加漏检概率。

#### 可能解决方法

1. 在高目标相似度区域，即使没有检测框，也进入独立的目标恢复流程。
2. 原地旋转获取多个视角，并使用同义词、复数形式、较低阈值和多尺度裁剪重试 GroundingDINO。
3. 复用 Qwen 对“目标是否清晰可见”进行语义确认。
4. 使用连续 2～3 帧证据融合，避免单帧 VLM 误判直接停止。
5. 停止条件同时要求语义确认和目标深度/导航距离位于成功半径内。
6. 在调用 Open3D 前检查空点云；同时确认 `object_map_erosion_size` 使用合理值（通常约为 `5`，而不是 `1600`）。

### BENCH-003：楼梯漏检导致跨楼层策略无法触发

#### 现象

当目标位于其他楼层时，如果楼梯识别算法没有识别出楼梯区域，机器人只能在当前楼层继续探索，无法进入 `climb_stair`。

#### 当前机制缺口

当前楼梯地图主要依赖高精度硬融合：

```text
GroundingDINO/SAM stair mask
AND
RedNet stair pixels > 20
AND
两个 mask 存在交集
```

上楼梯地图还需要投影面积超过约 20 个像素才设置 `_has_up_stair=True`。任一感知模块漏检都会切断整个跨楼层链路。

另外，当前 OpenFrontier 语义策略只在已经进入 `climb_stair` 后运行，因此只能决定“楼梯上往哪里走”，不能解决“楼梯是否被发现”。

#### 可能解决方法

1. 将硬交集改成分级多源融合：
   - GroundingDINO/SAM 与 RedNet 同时命中：立即确认；
   - 单模型连续多帧稳定命中：中置信度确认；
   - 深度结构或平行台阶线提供弱候选，再交给 Qwen 确认。
2. 使用 hit-count 或概率楼梯地图累积时序证据，避免单帧漏检清除有效区域。
3. 在探索阶段增加低频楼梯语义确认入口，而不是等进入 `climb_stair` 后才调用语义模块。
4. 针对小目标、远距离楼梯和不同表述扩充提示词及检测类别。
5. 分开调节“楼梯候选召回阈值”和“正式触发跨楼层的确认阈值”。

### BENCH-004：楼梯区域太小导致过早退出爬楼模式

#### 现象

楼梯模型只检测到入口附近的小区域时，机器人很快到达 centroid。若下一帧没有及时扩展楼梯地图，机器人前进一步就会离开旧 mask，状态机错误认为已经走完楼梯并退出 `climb_stair`。

#### 已确认的触发链

```text
小面积楼梯 mask
→ centroid 位于楼梯入口附近
→ reach_stair_centroid=True
→ 当前帧新楼梯区域尚未写入地图
→ 机器人离开上一帧旧 mask
→ 状态机将“离开 mask”直接解释为爬楼成功
```

当前代码先使用旧 stair map 处理爬楼状态，之后才调用 `update_map` 融合当前帧观测；而且楼梯膨胀只在第一次状态处理时生效，后续又直接使用原始区域。

#### 当前已实现的缓解机制

评测配置已将 Habitat episodic GPS 设置为三维，并使用其竖直分量建立高度
时序保护。地图显示机器人离开楼梯区域时：

```text
上楼且最近高度仍上升 → 保持 climb_stair
下楼且最近高度仍下降 → 保持 climb_stair
高度变化稳定 + 地图外连续确认 6 step → 允许退出
```

默认使用最近 5 step 的高度窗口，方向性高度变化阈值为 `0.03m`。高度只
参与楼梯状态判断，不写入每层 obstacle map 的固定高度坐标系，避免破坏
上层地图的障碍物高度过滤。二维 GPS 配置无法提供高度时自动回退为原始
地图判定。

可调参数：

```bash
export ASCENT_STAIR_HEIGHT_WINDOW_STEPS=5
export ASCENT_STAIR_HEIGHT_CHANGE_THRESHOLD=0.03
export ASCENT_STAIR_EXIT_STABLE_STEPS=6
```

#### 后续可继续改进

1. 调整顺序：先融合当前帧楼梯检测，再判断机器人是否离开楼梯。
2. 爬楼期间使用历史累计楼梯地图与当前地图的并集，并持续进行适度膨胀。
3. “离开 stair mask”不能单独代表成功，还应至少满足：
   - 距离楼梯起点超过最小爬行距离；
   - 累计向前产生足够进展；
   - 检测到平台/楼梯终点；
   - 或语义模块确认已经到达目标楼层的出口方向。
4. 将“成功离开楼梯”和“在楼梯上卡住”设计为两个独立状态，避免使用同一个 paused 计数推断两种相反结果。

## 7. 推荐实施优先级

### P0：目标与状态生命周期修复

1. 修复唯一探索 frontier 绕过卡住检测的问题。
2. 使用有方向的距离进展和空间半径黑名单。
3. 先更新 stair map，再运行爬楼退出判断。
4. 增加楼梯退出滞后、最短爬行距离和保护步数。
5. 无有效语义楼梯候选时也更新检测时间，避免每步调用 Qwen。

这些修改不需要重新训练模型，对现有 PointNav 干预最小。

### P1：提高感知召回率

1. 增加目标漏检后的主动观察和 Qwen 恢复确认。
2. 将楼梯硬交集改为多源时序融合。
3. 增加探索阶段的楼梯弱候选和语义确认。
4. 处理空目标点云，避免 Open3D 对空数据聚类。

### P2：Benchmark 与消融实验

建议分别统计：

- 不可达 frontier 比例、平均禁用延迟和浪费步数。
- 进入目标成功距离后的检测召回率与恢复识别成功率。
- 楼梯检测 recall、楼梯模式触发率和实际通过率。
- `reach_stair_centroid` 后的过早退出率。
- FrontierNet 候选有效率、Qwen 低概率拒绝率和胡萝卜回退率。
- 每个 episode 的 FrontierNet/Qwen 调用次数和推理时间。

## 8. 失败 Episode 建议记录字段

每个失败 episode 至少保存：

- scene ID、episode ID、目标类别和目标所在楼层。
- 失败类型：BENCH-001～BENCH-004。
- 失败发生 step、机器人位姿、当前模式和楼梯方向。
- 常规探索 frontier、语义楼梯 frontier、被禁用目标及过滤原因。
- 目标检测框、SAM mask、RedNet 楼梯 mask 和累计楼梯地图。
- 是否到达楼梯 centroid、是否离开楼梯 mask、是否完成楼层切换。
- 目标相似度、目标点云数量、是否出现 Open3D 空点云警告。
- Success、SPL、SoftSPL、最终步数及对应 debug 视频路径。

## 9. 当前实现边界

当前语义楼梯策略已经完成：

- 官方 OpenFrontier/FrontierNet 接入；
- OpenCV 到 ASCENT 相机坐标修正；
- 复用多模态 Qwen 的上楼/下楼语义评分；
- 概率、距离、朝向和卡住目标过滤；
- 单一短程子目标输出及原胡萝卜回退；
- RGB、Set-of-Marks 和 top-down map 可视化；
- 坐标、候选选择、低概率回退、hysteresis 和可视化测试。

尚未完成的主要工作是常规 frontier 可达性验证、目标漏检恢复、楼梯发现阶段的多源融合，以及爬楼结束状态机的时序修正。这些问题应作为后续 benchmark 优化的主要方向。
