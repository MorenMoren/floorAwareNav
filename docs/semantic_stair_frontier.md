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
5. returns the best candidate as a short-range XY PointNav goal.

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
export ASCENT_STAIR_REPLAN_INTERVAL=4
export ASCENT_STAIR_MIN_SUBGOAL_DISTANCE=0.3
export ASCENT_STAIR_MAX_SUBGOAL_DISTANCE=2.5
export ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY=0.15
export ASCENT_STAIR_SEMANTIC_POWER=2.0
export ASCENT_STAIR_GAIN_POWER=1.0
export ASCENT_STAIR_HYSTERESIS_RATIO=1.2
export ASCENT_STAIR_STUCK_STEPS=8
```

Candidates scored by the VLM below
`ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY` are rejected. If a fresh FrontierNet
run has no acceptable candidate, navigation immediately falls back to the
legacy carrot instead of continuing toward a stale semantic goal. The console
prints every detected label and its filter reason, followed by the one selected
PointNav subgoal (if any).

Set `ASCENT_SEMANTIC_STAIR_ENABLED=0` to force the legacy carrot strategy.

## Tests

The policy selection, caching, filtering, prompt, parsing, and fallback tests do
not require FrontierNet or a VLM:

```bash
python -m unittest discover -s tests -p 'test_semantic_stair_frontier_policy.py' -v
```
