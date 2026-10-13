# Run app.vjepa_hanoi.paper_protocol_eval unchanged except for memory on a 16 GB GPU: CEM's samples go through the
# predictor 200 at a time (goal_planner.chunked: identical per-sample math) and the frozen weights do not require grad
# (no autocast weight cache). All arguments pass through to paper_protocol_eval.
import sys

sys.path.insert(0, "/home/cpw/workspace/vjepa2")

import app.vjepa_hanoi.paper_protocol_eval as ppe  # noqa: E402
from app.vjepa_hanoi.goal_planner import chunked  # noqa: E402

_make, _build = ppe.make_step_fn, ppe.build_models


def build_models(cfg, device):
    encoder, predictor = _build(cfg, device)
    encoder.requires_grad_(False)
    predictor.requires_grad_(False)
    return encoder, predictor


ppe.build_models = build_models
ppe.make_step_fn = lambda *a, **k: chunked(_make(*a, **k), 200)
sys.argv = ["paper_protocol_eval"] + sys.argv[1:]
ppe.main()
