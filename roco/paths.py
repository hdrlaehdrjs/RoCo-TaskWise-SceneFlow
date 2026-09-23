import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = Path(os.environ.get("ROCO_THIRD_PARTY", REPO / "third_party"))
CHECKPOINTS = Path(os.environ.get("ROCO_CHECKPOINTS", REPO / "checkpoints"))
SPRING = Path(os.environ.get("ROCO_SPRING", REPO / "data" / "spring"))

DEFOM_ROOT = THIRD_PARTY / "DEFOM-Stereo"
FLOW_LIBRARY = THIRD_PARTY / "flow_library"

DEFOM_INIT = CHECKPOINTS / "defom-stereo" / "defomstereo_vits_sceneflow.pth"
DPFLOW_INIT = CHECKPOINTS / "dpflow" / "dpflow-spring-69bac7fa.ckpt"

for path in (DEFOM_ROOT, DEFOM_ROOT / "core", FLOW_LIBRARY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
