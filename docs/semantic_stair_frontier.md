# Semantic stair frontier policy

The semantic stair policy replaces only the short-range carrot target generated
inside `Ascent_Policy._climb_stair`. Stair detection, PointNav actions,
completion detection, and floor-map switching remain unchanged.

At runtime the policy:

1. asks FrontierNet for visual-frontier candidates from the current RGB-D frame;
2. anchors candidates in the episodic coordinate frame;
3. marks candidates in the image;
4. asks a multimodal VLM which candidate safely continues up or down the current
   staircase;
5. locks the best candidate as a short-range XY PointNav goal until the robot
   enters its 0.8 m arrival radius.

While a selected semantic subgoal is farther than 0.8 m away, the policy
returns exactly the cached world-space target on every step. It does not call
FrontierNet or the VLM again, does not replace the target after four steps, and
does not fall back to the camera-relative carrot because progress is briefly
slow. Once the robot is within 0.8 m, the local segment is complete. Upstairs
traversal then returns the camera to level before a fresh RGB-D observation may
produce the next stair frontier. Candidates already inside the arrival radius
are filtered because they do not define a useful navigation segment.

The legacy centroid pause counter is held at zero while a semantic subgoal is
locked. Semantic traversal instead logs the absolute distance and signed
progress to the selected world-space goal. This prevents Ascent's old 30-step
centroid timeout from ending the climb before PointNav reaches the selected
frontier, and prevents a later stair-mask exit from being classified as a
timeout failure.

Upstairs traversal uses a level-detect/look-up-navigate cycle:

1. after reaching the stair centroid, return the camera to pitch zero;
2. run FrontierNet and Qwen only from that level view;
3. after selecting and caching a frontier, issue one `LOOK_UP` action;
4. navigate with PointNav while retaining the same world-space frontier;
5. inside the 0.8 m arrival radius, release it and return to pitch zero before
   detecting the next segment.

Downstairs pitch behavior remains unchanged. Leaving stair mode also clears
the semantic target and returns camera pitch toward zero through Ascent's
existing exit behavior.

If FrontierNet, its weights, the VLM, or a valid candidate is unavailable,
Ascent automatically uses the original carrot strategy.

When Habitat debug video is enabled, the top-down map overlays the latest
OpenFrontier stair candidates during stair traversal: cyan circles are valid
candidates, orange circles were rejected by geometric filters, and the filled
magenta circle is the active semantic stair subgoal. Candidate labels match
the Set-of-Marks labels sent to the VLM.

The same marked FrontierNet RGB image is shown in the debug video's VLM-input
panel while stair traversal is active. The accompanying text reports the
selected label, semantic probability, utility, and VLM reason.

For visibility, those candidates are also inverse-projected from FrontierNet's
center-cropped input and drawn directly on the debug video's main RGB view.

## Setup

Use an official OpenFrontier checkout rather than copying its source into this
repository:

```bash
git clone --recursive https://github.com/cvg/OpenFrontier.git third_party/OpenFrontier
pip install -r requirements_stair_frontier.txt
cd third_party/OpenFrontier
bash scripts/download_weights.sh
cd ../..
```

The default paths are:

```text
OPENFRONTIER_ROOT=third_party/OpenFrontier
OPENFRONTIER_WEIGHTS=third_party/OpenFrontier/model_weights/rgbd_11cls.pth
```

The weights variable is optional when the checkpoint exists at the default
OpenFrontier path.

The stair scorer reuses the same multimodal Qwen3.6 client instance as
`Ascent_LLM_Planner`; it does not launch a second VLM or require a separate API
key. Configure the existing OpenAI-compatible endpoint as usual:

```bash
export ASCENT_QWEN_MODEL=Qwen/Qwen3.6-27B
export ASCENT_QWEN_BASE_URL=http://your-qwen-server/v1
export ASCENT_QWEN_API_KEY=EMPTY
```

## Main controls

```bash
export ASCENT_SEMANTIC_STAIR_ENABLED=1
export ASCENT_STAIR_GOAL_REACHED_DISTANCE=0.8
export ASCENT_STAIR_MIN_SUBGOAL_DISTANCE=0.3
export ASCENT_STAIR_MAX_SUBGOAL_DISTANCE=2.5
export ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY=0.15
export ASCENT_STAIR_SEMANTIC_POWER=2.0
export ASCENT_STAIR_GAIN_POWER=1.0
export ASCENT_STAIR_HYSTERESIS_RATIO=1.2
export ASCENT_STAIR_HEIGHT_WINDOW_STEPS=5
export ASCENT_STAIR_HEIGHT_CHANGE_THRESHOLD=0.03
export ASCENT_STAIR_EXIT_STABLE_STEPS=6
```

`ASCENT_STAIR_REPLAN_INTERVAL` and `ASCENT_STAIR_STUCK_STEPS` are retained in
the configuration for backward compatibility, but they do not release or
replace an active locked semantic stair subgoal.

HM3D and MP3D evaluation configs expose Habitat's episodic GPS in three
dimensions. Its vertical component is used only for stair-state decisions;
ASCENT's per-floor map projection retains its original fixed local camera
height. When the 2D obstacle map temporarily reports that the robot is outside
the stair mask, the state machine remains in `climb_stair` while recent height
changes in the requested direction (rising upstairs or falling downstairs).
Exit is confirmed only after the directional height change drops below 3 cm
over the five-step window and the map remains outside for six stable steps.
With a legacy 2D GPS configuration, this guard is unavailable and behavior
falls back to the original map-only decision.

Candidates scored by the VLM below
`ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY` are rejected. If a fresh FrontierNet
run has no acceptable candidate, navigation immediately falls back to the
legacy carrot instead of continuing toward a stale semantic goal. The console
prints every detected label and its filter reason, followed by the one selected
PointNav subgoal (if any).

Empty and fully rejected FrontierNet results also update the replan timestamp.
During legacy fallback, detection is therefore retried at the configured
`ASCENT_STAIR_REPLAN_INTERVAL` instead of every step, and a repeated empty
detection does not reset the legacy paused-step recovery counter.

Set `ASCENT_SEMANTIC_STAIR_ENABLED=0` to force the legacy carrot strategy.

## Tests

The policy selection, caching, filtering, prompt, parsing, and fallback tests do
not require FrontierNet or a VLM:

```bash
python -m unittest discover -s tests -p 'test_semantic_stair_frontier_policy.py' -v
```
