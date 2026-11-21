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
OUTPUT = ROOT / "output/m-instance-expanded"
SCREEN = OUTPUT / "data/screening.json"
EPOCHS = {"task1": 8, "increment": 3, "finetune": 1}
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
TASK1_IMAGES = 6144
TASK2_IMAGES = 1536
ARMS = ("matched", "off", "shuffled")
PRE_EVAL_FIX = "41f42c12d9a64911291ca04e56df102bc5759ce6"
FIXED_EVAL_BLOB = "675d49ed86e58a63a5522d04fdfb5fc94e22616e"


def command(arm, phase, parent=None, gnn=None, *, output=None, screen=None, epochs=None, seed=42,
            manifest=MANIFEST):
    output = OUTPUT if output is None else output
    screen = SCREEN if screen is None else screen
    stage = int(phase != "task1")
    folder = output / arm / phase
    epochs = EPOCHS[phase] if epochs is None else epochs
    lr = 5e-6 if phase == "finetune" else 5e-5
    image_root = resolve_image_root(manifest, read_json(manifest))
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
        "--expanded-instance-bank",
        "--instance-cache",
        str(output / "retrieval"),
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
        str(screen),
        "--owod-manifest",
        str(manifest),
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
        str(seed),
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


def calibrate(arm, checkpoint, *, output=None, screen=None, seed=42):
    output = OUTPUT if output is None else output
    screen = SCREEN if screen is None else screen
    folder = output / arm / "gnn"
    path = folder / "gnn_stage0.pt"
    if not (folder / "calibration_complete.json").exists() or not path.is_file():
        run(
            [
                sys.executable,
                str(ROOT / "tools/owod/calibrate_interference_gnn.py"),
                "--manifest",
                str(MANIFEST),
                "--owod-screening",
                str(screen),
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
                str(seed),
            ]
        )
    return path


def compare(rows, phase, arm, ref):
    left, right = rows[arm][phase], rows[ref][phase]
    return {key: left[key] - right[key] for key in METRICS if key in left and key in right}


def verify_or_create_plan(path, identity):
    if not path.exists():
        write_json(path, identity)
        return
    original = read_json(path)
    if original == identity:
        return
    if original != dict(identity, commit=PRE_EVAL_FIX):
        raise ValueError(f"Different run identity at {path}")

    # Only the direct repair of the failed evaluation can finish this old plan.
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    commit = identity["commit"]
    if (git("rev-parse", commit + "^") != PRE_EVAL_FIX
            or git("rev-parse", commit + ":engine.py") != FIXED_EVAL_BLOB
            or set(git("diff", "--name-only", PRE_EVAL_FIX, commit).splitlines())
            != {"engine.py", "tools/owod/run_instance_pilot.py"}
            or git("diff", "--name-only", commit)):
        raise ValueError("The checkout differs from the approved evaluation repair")

    import torch
    folder = path.parent / "matched/task1"
    checkpoint = folder / "checkpoint.pth"
    metadata = torch.load(checkpoint, map_location="meta", weights_only=False)
    if metadata["epoch"] != EPOCHS["task1"] - 1:
        raise ValueError("Expected the completed Task 1 training checkpoint for evaluation recovery")
    receipt = path.with_name("evaluation_recovery.json")
    recovery = dict(training_commit=PRE_EVAL_FIX, evaluation_commit=commit,
                    epoch=metadata["epoch"], checkpoint_sha256=file_sha256(checkpoint))
    if receipt.exists():
        if read_json(receipt) != recovery:
            raise ValueError("Task 1 evaluation recovery provenance changed")
    else:
        if (folder / "training_complete.json").exists():
            raise ValueError("Expected a Task 1 evaluation interruption")
        write_json(receipt, recovery)
    print(f"Verified Task 1 endpoint for evaluation recovery: {checkpoint}", flush=True)


def pilot():
    import torch

    if torch.cuda.device_count() != 2 or any("3090" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Select two RTX 3090 devices with CUDA_VISIBLE_DEVICES")
    from models.graph_local.instance_transfer import bank_settings
    from tools.owod.instance_coverage import task1_coverage

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
    screen = prepare_screen(MANIFEST, SCREEN.parent, TASK1_IMAGES, 128, 42, task2_images=TASK2_IMAGES)
    identity = dict(
        commit=commit,
        manifest=file_sha256(MANIFEST),
        screen=file_sha256(screen),
        weights=file_sha256(WEIGHTS),
        epochs=EPOCHS,
        instance_settings=bank_settings(expanded=True),
        task2_arms=list(ARMS),
        devices=os.environ["CUDA_VISIBLE_DEVICES"],
        gpus=[torch.cuda.get_device_name(i) for i in range(2)],
        torch_version=str(torch.__version__),
        batch_per_device=2,
        increment_images=[TASK1_IMAGES, TASK2_IMAGES],
        seed=42,
        evaluation="official full M test, VOC integral AP50; fixed endpoints",
        finetune="entire author balanced memory",
        paper_comparable=False,
    )
    path = OUTPUT / "plan.json"
    verify_or_create_plan(path, identity)
    rows = {}
    deltas = {}
    report = dict(
        plan=str(path),
        rows=rows,
        deltas=deltas,
        units="fractions; counts for unknown_predictions and a_ose",
    )
    parent, result = train("matched", "task1")
    parent_dir = parent.parent
    audit = parent_dir / "candidate_coverage.json"
    if not audit.is_file():
        write_json(audit, task1_coverage(parent, MANIFEST, SCREEN))
    report["task1"] = dict(metrics=result, coverage=read_json(audit), checkpoint=str(parent))
    report["common_parent_sha256"] = file_sha256(parent)
    write_json(OUTPUT / "comparison.json", report)
    gnn = calibrate("matched", parent)
    report["common_gnn_sha256"] = file_sha256(gnn)
    for arm in ARMS:
        inc, increment = train(arm, "increment", parent, gnn)
        layout = read_json(inc.parent / "graph_adapter_config.json")["specs"][-1]
        evidence = layout["instances"]
        if (layout["source_checkpoint_sha256"] != report["common_parent_sha256"]
                or layout["gnn_sha256"] != report["common_gnn_sha256"]):
            raise ValueError(f"{arm} did not use the planned Task 1 parent and GNN")
        if arm == "matched":
            report["evidence_sha256"] = evidence["cache_sha256"]
            representatives = [entry for slot in evidence["representatives"] for entry in slot]
            report["bank"] = dict(images_scanned=evidence["bank_images_scanned"],
                                  occupied_slots=len(evidence["representatives"]),
                                  representatives=len(representatives),
                                  source_images=len({entry["image"] for entry in representatives}))
        elif evidence["cache_sha256"] != report["evidence_sha256"]:
            raise ValueError(f"{arm} did not use the shared frozen-parent evidence")
        rows[arm] = dict(increment=increment, retrieval=layout["alignment_summary"])
        write_json(OUTPUT / "comparison.json", report)
        _, rows[arm]["finetune"] = train(arm, "finetune", inc, gnn)
        if arm != "matched":
            deltas[f"matched_minus_{arm}"] = {
                phase: compare(rows, phase, "matched", arm) for phase in ("increment", "finetune")
            }
        write_json(OUTPUT / "comparison.json", report)

    active = rows["matched"]["increment"]["mean_alignment_count_per_rank_batch"]
    if not active or not rows["shuffled"]["retrieval"]["shuffled_across_slots"]:
        report["transfer_conclusion"] = "Inconclusive: insufficient training exposure or distinct shuffled targets"
    else:
        improved = all(deltas["matched_minus_" + ref]["finetune"]["current_ap50"] > 0
                       for ref in ("off", "shuffled"))
        report["transfer_conclusion"] = ("Higher final Current AP50 than both controls in this pilot"
            if improved else "No consistent final Current AP50 advantage over both controls")
    report["status"] = "complete"
    write_json(OUTPUT / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="show the schedule without training")
    if parser.parse_args().plan:
        from models.graph_local.instance_transfer import bank_settings
        bank = bank_settings(expanded=True)
        print(f"M-OWODB, 2 x RTX 3090, batch 2 per GPU. One Task 1 parent: {TASK1_IMAGES} images x {EPOCHS['task1']} epochs.")
        print(f"Bank: {bank['bank_images']} images, {bank['slots']} slots x {bank['per_slot']} representatives.")
        print(f"Task 2: {', '.join(ARMS)}; each {TASK2_IMAGES} images x {EPOCHS['increment']} epochs + full author memory x {EPOCHS['finetune']}.")
        print(f"Full M author evaluation at phase endpoints. Output: {OUTPUT}")
    else:
        import fcntl

        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / "pilot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pilot()
