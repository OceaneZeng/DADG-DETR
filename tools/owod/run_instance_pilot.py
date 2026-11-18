"""Fixed-budget M-OWODB Task 1 -> Task 2 comparison on two 3090s."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod.protocol import file_sha256, read_json, resolve_image_root
from tools.owod.screening_data import prepare_screen, write_json

MANIFEST = ROOT / "data/owod/m-owodb/split_manifest.json"
WEIGHTS = ROOT / "pretrained/dino_resnet50.pth"
OUTPUT = ROOT / "output/m-instance-pilot"
SCREEN = OUTPUT / "data/screening.json"
EPOCHS = {"task1": 4, "increment": 3, "finetune": 1}
METRICS = (
    "previous_ap50",
    "current_ap50",
    "known_ap50",
    "u_recall",
    "unknown_precision",
    "unknown_predictions",
    "wi",
    "a_ose",
    "h_score",
)
TRAIN_IMAGES = 1536


def command(arm, phase, parent=None, gnn=None):
    stage = int(phase != "task1")
    folder = OUTPUT / arm / phase
    epochs = EPOCHS[phase]
    lr = 5e-6 if phase == "finetune" else 5e-5
    image_root = resolve_image_root(MANIFEST, read_json(MANIFEST))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(ROOT / "main.py"),
        "--instance-mode",
        arm,
        "--instance-cache",
        str(OUTPUT / "retrieval"),
        "--owod-discovery",
        "--graph-routed-adapter",
        "--graph-routing",
        "soft",
        "--graph-localization",
        "frozen",
        "--graph-sketch-max-images",
        "8",
        "--discovery-scoring",
        "focal",
        "--discovery-known-rank-coef",
        "1",
        "--discovery-pseudo-ramp-epochs",
        "3",
        "--discovery-pseudo-normalization",
        "count",
        "--activation-checkpointing",
        "--screen-author-eval",
        "--owod-screening",
        str(SCREEN),
        "--owod-manifest",
        str(MANIFEST),
        "--owod-recipe",
        "ow-detr-paper",
        "--owod-stage",
        str(stage),
        "--owod-phase",
        "finetune" if phase == "finetune" else "increment",
        "--output_dir",
        str(folder),
        "--coco_path",
        str(image_root),
        "--num_classes",
        "91",
        "--num_queries",
        "100",
        "--dim_feedforward",
        "1024",
        "--batch_size",
        "2",
        "--num_workers",
        "2",
        "--backbone-weights",
        str(WEIGHTS),
        "--seed",
        "42",
        "--epochs",
        str(epochs),
        "--eval_interval",
        str(epochs),
        "--lr_drop",
        str(epochs),
        "--lr",
        str(lr),
        "--lr_backbone",
        str(lr / 10),
        "--weight_decay",
        ".0001",
    ]
    if stage:
        cmd += ["--pretrained", str(parent), "--graph-gnn-checkpoint", str(gnn)]
    return cmd


def run(cmd):
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def metrics(folder, epoch):
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line]
    final = [r for r in rows if r["epoch"] == epoch and "test_owod_known_ap50" in r]
    if len(final) != 1:
        raise ValueError(f"Missing or duplicate final evaluation: {folder}")
    values = {}
    for key in METRICS:
        value = final[0].get("test_owod_" + key)
        if key == "previous_ap50" and folder.name == "task1":
            continue
        if value is None or not math.isfinite(value):
            raise ValueError(f"Missing {key} in {folder}")
        values[key] = value
    values["mean_alignment_count_per_rank_batch"] = final[0].get("train_discovery_alignment_count")
    return values


def train(arm, phase, parent=None, gnn=None):
    folder = OUTPUT / arm / phase
    checkpoint = folder / "checkpoint.pth"
    marker = folder / "training_complete.json"
    if not marker.exists():
        cmd = command(arm, phase, parent, gnn)
        if checkpoint.exists():
            cmd += ["--resume", str(checkpoint)]
        run(cmd)
    if read_json(marker)["last_epoch"] != EPOCHS[phase] - 1 or not checkpoint.is_file():
        raise ValueError(f"Incomplete {folder}")
    return checkpoint, metrics(folder, EPOCHS[phase] - 1)


def calibrate(arm, checkpoint):
    folder = OUTPUT / arm / "gnn"
    path = folder / "gnn_stage0.pt"
    if not (folder / "calibration_complete.json").exists() or not path.is_file():
        run(
            [
                sys.executable,
                str(ROOT / "tools/owod/calibrate_interference_gnn.py"),
                "--manifest",
                str(MANIFEST),
                "--owod-screening",
                str(SCREEN),
                "--stage",
                "0",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(folder),
                "--num_workers",
                "0",
                "--sketch-max-images",
                "4",
                "--probe-max-images",
                "4",
                "--probe-steps",
                "3",
                "--seed",
                "42",
            ]
        )
    return path


def compare(rows, phase, arm, ref):
    left, right = rows[arm][phase], rows[ref][phase]
    return {key: left[key] - right[key] for key in METRICS if key in left and key in right}


def pilot():
    import torch

    if torch.cuda.device_count() != 2 or any("3090" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Select two RTX 3090 devices with CUDA_VISIBLE_DEVICES")
    from models.graph_local.instance_transfer import SETTINGS

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run(
        [
            sys.executable,
            str(ROOT / "tools/owod/manage_data.py"),
            "verify",
            "--benchmark",
            "m-owodb",
            "--output-root",
            str(ROOT / "data/owod"),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            str(ROOT / "tools/owod/server_cuda_check.py"),
            "--instances",
        ]
    )
    screen = prepare_screen(MANIFEST, SCREEN.parent, TRAIN_IMAGES, 128, 42)
    identity = dict(
        commit=commit,
        manifest=file_sha256(MANIFEST),
        screen=file_sha256(screen),
        weights=file_sha256(WEIGHTS),
        epochs=EPOCHS,
        instance_settings=SETTINGS,
        devices=os.environ["CUDA_VISIBLE_DEVICES"],
        gpus=[torch.cuda.get_device_name(i) for i in range(2)],
        torch_version=str(torch.__version__),
        batch_per_device=2,
        train_images=TRAIN_IMAGES,
        seed=42,
        evaluation="official full M test, VOC integral AP50; fixed endpoints",
        finetune="entire author balanced memory",
        paper_comparable=False,
    )
    path = OUTPUT / "plan.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError(
            f"Different run identity at {path}; preserve this run and use a fresh output directory"
        )
    write_json(path, identity)
    rows = {}
    deltas = {}
    report = dict(
        plan=str(path),
        rows=rows,
        deltas=deltas,
        units="fractions; counts for unknown_predictions and a_ose",
    )
    for parent_arm in ("legacy", "matched"):
        t1, result = train(parent_arm, "task1")
        rows[parent_arm] = {"task1": result}
        from tools.owod.instance_coverage import task1_coverage

        audit = OUTPUT / parent_arm / "task1" / "candidate_coverage.json"
        if not audit.is_file():
            write_json(audit, task1_coverage(t1, MANIFEST, SCREEN))
        rows[parent_arm]["task1_coverage"] = read_json(audit)
        write_json(OUTPUT / "comparison.json", report)
        gnn = calibrate(parent_arm, t1)
        arms = ("legacy",) if parent_arm == "legacy" else ("matched", "off", "shuffled")
        for arm in arms:
            if arm != parent_arm:
                if not report["retrieval"]["accepted"]:
                    report["transfer_conclusion"] = "No accepted instances; transfer effect unmeasured"
                    break
                rows[arm] = {}
            inc, rows[arm]["increment"] = train(arm, "increment", t1, gnn)
            if arm != "legacy":
                layout = read_json(OUTPUT / arm / "increment/graph_adapter_config.json")["specs"][-1]
                summary = layout["alignment_summary"]
                if arm == "matched":
                    report["retrieval"] = summary
                    report["common_parent_sha256"] = layout["source_checkpoint_sha256"]
                    report["evidence_sha256"] = layout["instances"]["cache_sha256"]
                elif (
                    layout["source_checkpoint_sha256"] != report["common_parent_sha256"]
                    or layout["instances"]["cache_sha256"] != report["evidence_sha256"]
                ):
                    raise ValueError("Paired arms used different parent or evidence")
                if arm == "shuffled":
                    report["shuffled_across_slots"] = summary["shuffled_across_slots"]
            _, rows[arm]["finetune"] = train(arm, "finetune", inc, gnn)
            if arm != "legacy":
                ref, method = ("legacy", "matched") if arm == "matched" else (arm, "matched")
                deltas[f"{method}_minus_{ref}"] = {
                    phase: compare(rows, phase, method, ref) for phase in ("increment", "finetune")
                }
            write_json(OUTPUT / "comparison.json", report)
    deltas["matched_minus_legacy"]["task1"] = compare(rows, "task1", "matched", "legacy")
    if "shuffled" in rows:
        active = rows["matched"]["increment"]["mean_alignment_count_per_rank_batch"]
        if not active or not report["shuffled_across_slots"]:
            report["transfer_conclusion"] = (
                "Inconclusive: insufficient training exposure or distinct shuffled targets"
            )
        else:
            improved = all(
                deltas["matched_minus_" + ref]["finetune"]["current_ap50"] > 0
                for ref in ("off", "shuffled")
            )
            report["transfer_conclusion"] = (
                "Higher final Current AP50 than both controls in this pilot"
                if improved
                else "No consistent final Current AP50 advantage over both controls"
            )
    report["status"] = "complete"
    write_json(OUTPUT / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="show the schedule without training")
    if parser.parse_args().plan:
        print(f"M-OWODB, 2 x RTX3090, {TRAIN_IMAGES} increment images per task; {EPOCHS}")
        print(
            "Task1 legacy/matched; Task2 legacy and shared-parent matched/off/shuffled; full author evaluation."
        )
    else:
        import fcntl

        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / "pilot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pilot()
