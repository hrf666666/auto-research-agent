---
name: dataset-understanding
description: "Thoroughly understand and validate all datasets before any experiment. Produces a canonical DATASET_MANIFEST.json by inspecting actual files on disk, never by assumption."
---

# /dataset-understanding

**This skill is MANDATORY on first cycle and whenever `data/` changes.**

## Core Principle

**NEVER assume. ALWAYS verify with actual file reads.**

## How to Execute

You MUST write and run a Python script that performs ALL checks below, then writes `workspace/DATASET_MANIFEST.json`. Do NOT manually inspect files and guess — write code that programmatically checks everything.

### Step 1: Create and Run Inspection Script

Write a script `scripts/_inspect_datasets.py` that does the following, then RUN it:

```python
"""Dataset inspection script — produces DATASET_MANIFEST.json from actual files."""
import os, json, glob, numpy as np
from pathlib import Path
from PIL import Image

DATA_ROOT = "data"
MANIFEST_PATH = "workspace/DATASET_MANIFEST.json"
manifest = {"version": "", "datasets": {}, "excluded": {}, "scenes_without_gt": {}, "total_trainable": {"train": 0, "val": 0, "all": 0}}

def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        dims = f.readline().decode().strip().split()
        scale = float(f.readline().decode().strip())
        w, h = int(dims[0]), int(dims[1])
        data = np.fromfile(f, dtype=np.float32, count=h*w).reshape((h, w))
        if scale < 0: data = np.flipud(data)
    return data

def check_dir(base, dataset_name, type_name):
    """Generic dataset checker. Returns (train_scenes, val_scenes, info_dict)."""
    train, val, no_gt = [], [], []
    for scene_dir in sorted(Path(base).iterdir()):
        if not scene_dir.is_dir(): continue
        # --- Find input images (any of the known patterns) ---
        input_count = 0
        input_pattern = ""
        if list(scene_dir.glob("input_Cam000.png")):
            input_count = len(list(scene_dir.glob("input_Cam*.png")))
            input_pattern = "input_Cam{i:03d}.png"
        elif list(scene_dir.glob("1_1.png")):
            input_count = len(list(scene_dir.glob("*.png"))) - len(list(scene_dir.glob("*_disparity.npy"))) - len(list(scene_dir.glob("*_depth.npy")))
            input_pattern = "{row}_{col}.png"
        elif (scene_dir / "color").is_dir() or (scene_dir / "Color").is_dir():
            cdir = scene_dir / "color" if (scene_dir / "color").is_dir() else scene_dir / "Color"
            input_count = len(list(cdir.glob("*.png")))
            input_pattern = "color/*.png"
        if input_count == 0: continue

        # --- Find depth GT (try ALL known patterns) ---
        gt_info = None
        # Pattern A: gt_disp_lowres.pfm / gt_depth_lowres.pfm
        for gf in ["gt_disp_lowres.pfm", "gt_disp_highres.pfm", "gt_depth_lowres.pfm", "gt_depth_highres.pfm"]:
            gp = scene_dir / gf
            if gp.exists():
                d = read_pfm(str(gp))
                gt_info = {"format": "pfm", "pattern": gf, "shape": list(d.shape), "range": [float(d.min()), float(d.max())]}
                break
        # Pattern B: *_disparity.npy (UrbanLF-Syn per-view)
        if gt_info is None:
            disp_files = list(scene_dir.glob("*_disparity.npy"))
            if disp_files:
                d = np.load(str(disp_files[0]))
                gt_info = {"format": "npy_disparity", "pattern": disp_files[0].name, "shape": list(d.shape), "range": [float(d.min()), float(d.max())]}
        # Pattern C: disp_*.npy (Non-lambertian)
        if gt_info is None:
            disp_files = list(scene_dir.glob("disp_*.npy"))
            if disp_files:
                d = np.load(str(disp_files[0]))
                gt_info = {"format": "npy_disparity", "pattern": disp_files[0].name, "shape": list(d.shape), "range": [float(d.min()), float(d.max())]}
        # Pattern D: depth/*.png or Depth/*.png (Wanner)
        if gt_info is None:
            for dname in ["depth", "Depth"]:
                dd = scene_dir / dname
                if dd.is_dir():
                    pngs = sorted(dd.glob("*.png"))
                    if pngs:
                        d = np.array(Image.open(str(pngs[0])), dtype=np.float32)
                        if d.max() - d.min() > 1.0:  # NOT all zeros
                            gt_info = {"format": "depth_png_uint8", "pattern": f"{dname}/*.png", "shape": list(d.shape), "range": [float(d.min()), float(d.max())]}
                        else:
                            no_gt.append({"scene": scene_dir.name, "reason": f"{dname} values all zero ({float(d.min())},{float(d.max())})"})
                        break
        # Pattern E: r_map.npy — this is REFLECTANCE, NOT depth GT
        if gt_info is None:
            rmap = list(scene_dir.glob("r_map.npy"))
            if rmap:
                d = np.load(str(rmap[0]))
                no_gt.append({"scene": scene_dir.name, "reason": f"Only r_map.npy (reflectance {float(d.min()):.3f}-{float(d.max()):.3f}), no depth GT"})

        if gt_info is None and scene_dir.name not in [x["scene"] for x in no_gt]:
            no_gt.append({"scene": scene_dir.name, "reason": "No recognized GT file found"})

        if gt_info:
            train.append({"scene": scene_dir.name, "gt": gt_info, "input_count": input_count, "input_pattern": input_pattern})

    return train, val, no_gt

# --- Inspect each dataset ---
datasets_found = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith(".")])
print(f"Found directories in data/: {datasets_found}")

for ds_dir in datasets_found:
    ds_path = os.path.join(DATA_ROOT, ds_dir)
    print(f"\n=== Inspecting {ds_dir} ===")

    # Check if this looks like a dataset or just data files
    has_subdirs = any(os.path.isdir(os.path.join(ds_path, x)) for x in os.listdir(ds_path))
    has_images = bool(glob.glob(os.path.join(ds_path, "**/*.png"), recursive=True)) or bool(glob.glob(os.path.join(ds_path, "**/*.h5"), recursive=True))

    if not has_images and not glob.glob(os.path.join(ds_path, "*.h5")):
        manifest["excluded"][ds_dir] = {"reason": "No image or data files found"}
        continue

    # Run generic checker on this directory
    # For HCInew, iterate sub-splits
    if ds_dir == "HCInew":
        # ... (specific handling for split-based datasets)
        pass
    # ... etc

# Write manifest
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest written to {MANIFEST_PATH}")
print(f"Datasets: {list(manifest['datasets'].keys())}")
print(f"Excluded: {list(manifest['excluded'].keys())}")
print(f"Scenes without GT: {list(manifest['scenes_without_gt'].keys())}")
```

**CRITICAL: The above is a TEMPLATE. You must adapt it to handle each specific dataset's directory structure. The script MUST:**

### Step 2: For Each Dataset, Determine Structure

Run these shell commands to understand the layout:

```bash
# List all datasets
ls data/

# For EACH dataset directory, show structure
find data/<dataset>/ -maxdepth 2 -type d | head -30

# Count scenes
ls data/<dataset>/*/
```

### Step 3: Validate GT For Every Scene

For EACH scene found, check:

1. **Does a GT file exist?** Try all patterns: `gt_*.pfm`, `*_disparity.npy`, `disp_*.npy`, `depth/*.png`, `Depth/*.png`
2. **Is the GT valid?** Load it and check:
   - Shape is 2D (H, W) or 3D with channel dim (1, H, W) — not 3D class map
   - Values are continuous — not discrete integers 1-11
   - Range is reasonable — not all zeros
   - `r_map.npy` → REFLECTANCE, not depth (exclude). Some datasets (e.g. Non-lambertian) have BOTH `r_map.npy` AND a real GT like `disp_*.npy`. Only count scenes with actual depth/disparity GT.
3. **If no valid GT** → add to `scenes_without_gt` list, mark status as `"NO GT"`

### Step 3.5: CRITICAL — Record Heterogeneous Resolution Info

Different datasets almost always have DIFFERENT image resolutions and GT formats. This is
the #1 cause of training crashes. You MUST:

1. **Record native resolution for EVERY scene** — both input images AND GT
2. **Summarize per-dataset**: native resolution, GT format, GT shape, GT value range
3. **Add a `resolution_notes` field** to each dataset entry listing the native resolution
4. **Add a `training_recommendations` section** at the manifest top level with:
   - Recommended `target_size` for unified training (pick the smallest common resolution or a standard one)
   - List of datasets that need resizing
   - Any datasets that CANNOT be loaded (broken format, h5, etc.) and must be excluded
5. **Mark scenes without usable GT as status containing "NO GT"** so the dataset loader skips them

Example `training_recommendations`:
```json
"training_recommendations": {
  "target_size": [480, 640],
  "num_views": 9,
  "exclude_datasets": ["Wanner_HCI_dataset", "HCI-Old"],
  "sampling_mode": "capped_balanced",
  "reason": "HCInew 512x512, Non-lambertian 926x926, UrbanLF-Syn 480x640. Wanner uses h5 format. Must unify to (480,640) for batching. Severe imbalance: use capped_balanced to boost minority domains."
}
```

The training script `train_v11.py` supports 3 `--sampling` modes:
- `natural`: all scenes shuffled equally. Good when domains are balanced.
- `balanced`: equal probability per domain. WARNING: over-upsamples tiny domains (1-2 scenes → 170x repetition → overfitting).
- `capped_balanced`: boosts minority domains but caps oversampling at 10x. RECOMMENDED for imbalanced datasets.

### Step 3.6: CRITICAL — Detect Data Quality Problems That Block Research Progress

Beyond basic format validation, you MUST detect and report these structural problems that
prevent meaningful training. For each problem found, add it to a `data_quality_issues` section
in the manifest with severity (`critical` / `high` / `medium`) and a recommended action.

**Problem 1: Severe Dataset Imbalance**

Count scenes per dataset per split. If any dataset has < 10% of the total scenes OR a
val split has < 3 scenes, this is a `high` severity problem.

```json
"data_quality_issues": {
  "dataset_imbalance": {
    "severity": "high",
    "detail": "UrbanLF-Syn has 200 scenes (91%) vs HCInew 16 (7%) vs Non-lambertian 6 (3%). Model will overfit to UrbanLF patterns.",
    "recommendation": "Use --sampling capped_balanced in train_v11.py. This boosts minority domains but caps oversampling at 10x to prevent overfitting. Pure balanced sampling WILL overfit when minority domains have < 5 scenes."
  }
}
```

**Problem 2: Missing Val Split for Entire Domain**

If any domain type (Lambertian / Non-Lambertian / Mixed) has ZERO validation scenes, this
is `critical` — you CANNOT measure per-domain generalization.

```json
"data_quality_issues": {
  "missing_val_domain": {
    "severity": "critical",
    "detail": "Non-Lambertian domain has 6 train scenes but 0 val scenes. Cannot evaluate Non-Lambertian generalization.",
    "recommendation": "Split Non-Lambertian: move 1-2 scenes (e.g. Teddy + David 80%) to val. This is REQUIRED for per-domain evaluation."
  }
}
```

**Problem 3: Shared GT Across Scenes (GT Duplication)**

If multiple scenes within the same dataset reference the same GT file (e.g. different
reflectance angles sharing one disparity map), flag this. The model may learn to ignore
input variations since GT is identical.

```json
"data_quality_issues": {
  "gt_duplication": {
    "severity": "medium",
    "detail": "5 David scenes in Non-lambertian share disp_david.npy (identical GT despite different reflectance). Model sees different inputs → same output.",
    "recommendation": "Deduplicate: use only 1 David scene for train and 1 for val, OR accept that these scenes test reflectance-invariance rather than depth diversity."
  }
}
```

**Problem 4: GT Value Range Inconsistency Across Datasets**

Load a sample GT from each dataset and compare value ranges. If one dataset's GT is in
[0, 1] and another is in [0, 10], the loss function will be dominated by the larger-range
dataset. Flag if max values differ by more than 3x across datasets.

```json
"data_quality_issues": {
  "gt_range_inconsistency": {
    "severity": "high",
    "detail": "HCInew GT range [0.2, 4.1] vs Non-lambertian [-0.41, 0.41] vs UrbanLF-Syn [0.0, 8.5]. Different scales will dominate loss.",
    "recommendation": "The dataset loader must normalize each GT independently (e.g. percentile-based to [0,1]) before computing loss. Verify the loader does this."
  }
}
```

**Problem 5: GT Format Shape Mismatch (2D vs 3D)**

Some .npy GT files are (H, W) and others are (1, H, W). If not handled, F.interpolate
or loss computation will crash or produce wrong results.

```json
"data_quality_issues": {
  "gt_shape_mismatch": {
    "severity": "high",
    "detail": "Non-lambertian disp_david.npy shape is (1, 256, 256) not (256, 256). Needs squeeze(0) before adding channel dim.",
    "recommendation": "Dataset loader must handle both 2D and 3D GT arrays: if ndim==3, squeeze leading dim first."
  }
}
```

**Implementation**: In your inspection script, after scanning all datasets, run a post-processing
step that checks for all 5 problems above and writes the `data_quality_issues` section.
This section MUST be present in the final manifest even if empty (use `{}` to indicate no issues).

### Step 4: Assign Splits

Based on directory structure:
- Subdirectories named `training/`, `additional/` → train
- Subdirectories named `stratified/`, `val/` → val
- Subdirectories named `test/` → test (but skip if no GT)

For datasets without sub-splits (Wanner, Non-lambertian):
- Use GT availability to determine inclusion
- Manual assignment for specific scenes when needed

### Step 5: Count and Verify

After generating the manifest, verify counts match:

```bash
# Verify scene counts
python3 -c "
import json
m = json.load(open('workspace/DATASET_MANIFEST.json'))
for ds, info in m['datasets'].items():
    total = info.get('total_scenes', 0)
    print(f'{ds}: {total} scenes')
print(f'No GT: {len(m[\"scenes_without_gt\"])} scenes')
"
```

### Step 6: Smoke Test the Unified Loader

After manifest is written, verify the loader agrees:

```python
from datasets.unified_lf_dataset import UnifiedLFDataset
ds = UnifiedLFDataset("data", target_size=128)
print(f"Loader: {len(ds)} scenes")
ds_stats = ds.get_stats()
print(f"Stats: {ds_stats}")

# Load one sample from each dataset type
seen = set()
for i in range(len(ds)):
    meta = ds._samples[i]
    if meta["dataset"] not in seen:
        seen.add(meta["dataset"])
        lf, depth, info = ds[i]
        print(f"  [{info['dataset']}] {info['scene']}: lf={lf.shape}, depth={depth.shape}")
```

If the loader count does NOT match the manifest → investigate and fix either the manifest or the loader.

### Step 7: Clean Up

After verification succeeds:
- DELETE `scripts/_inspect_datasets.py` (it's a one-time tool, not project code)
- Keep `workspace/DATASET_MANIFEST.json` as the permanent reference

## Output Format

The final `workspace/DATASET_MANIFEST.json` MUST contain these top-level keys:

```json
{
  "version": "YYYY-MM-DD",
  "generated_by": "dataset-understanding skill",
  "datasets": {
    "<dataset_name>": {
      "type": "Lambertian|Non-Lambertian|Mixed",
      "directory": "data/<dirname>",
      "input_format": "<format_tag>",
      "input_pattern": "<pattern>",
      "num_views": 81,
      "gt_format": "<format_tag>",
      "gt_pattern": "<pattern>",
      "gt_shape": [H, W],
      "native_resolution": [H, W],
      "total_scenes": N,
      "train_scenes": N,
      "val_scenes": N,
      "notes": "any important observations"
    }
  },
  "excluded": {
    "<dataset_name>": {"reason": "why excluded"}
  },
  "scenes_without_gt": {
    "<dataset/scene>": "reason"
  },
  "total_trainable": {"train": N, "val": N, "all": N},
  "training_recommendations": {
    "target_size": [H, W],
    "num_views": N,
    "exclude_datasets": ["<broken_dataset>"],
    "reason": "why these settings"
  },
  "data_quality_issues": {
    "dataset_imbalance": {
      "severity": "high|critical|medium",
      "detail": "description",
      "recommendation": "action to take"
    }
  }
}
```

## Anti-Patterns (MUST NOT DO)

1. Copy an old manifest without re-scanning files
2. Assume GT format from dataset name (HCI has both .pfm AND .h5)
3. Count license.txt as a scene
4. Include scenes with all-zero depth as valid
5. Use r_map.npy as depth GT
6. Skip checking case sensitivity (Color/ vs color/)
7. Only check the first scene — check ALL scenes
8. Ignore resolution differences between datasets — this WILL crash training when batching samples from different datasets
9. Forget to include `training_recommendations` with target_size — the training script needs to know what resolution to use
10. Assume all datasets have the same number of views or GT format
11. Ignore dataset imbalance — if one domain has 200 scenes and another has 6, the model will ignore the minority domain
12. Train on all domains without checking val split exists for each domain — you cannot evaluate what you cannot measure
13. Fail to detect GT duplication — if N scenes share the same GT file, the effective training diversity is N× less than expected
14. Assume GT normalization is consistent across datasets — always check value ranges and normalize per-scene
15. Ignore GT shape mismatches (2D vs 3D arrays) — this crashes loss computation
