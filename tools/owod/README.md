# Official M/S-OWODB workflow

See [the active author protocol](../../docs/owod-benchmarks.md).

1. `manage_data.py plan/check/prepare/verify`: own experiment data preparation and verification.
2. `prepare_benchmarks.py`: internal serializer for released ImageSets and original annotations.
3. `prepare_protocol.py`: register already-serialized author annotations with an image root.
4. `run_benchmark.py`: preflight and run four tasks with separate balanced finetuning.

The released lists and their pinned hashes are embedded in
`data/owod/{m-owodb,s-owodb}/split_manifest.json`.
Custom/pilot manifests and unverified overrides are rejected. Historical setting
files and launchers have been deleted. Read-only historical reporting lives in
`tools/analysis/`; model components remain in `models/`.

See [the data guide](../../docs/data-preparation.md) for raw sources and commands.

`download_data.py --benchmark both` resumes official COCO/ORE sources and DINO
backbone weights. It records sources/hashes and safely extracts ZIPs; ORE Drive
access requires gdown. An empty/unavailable author folder is a failure, not permission
to reconstruct the M annotations heuristically.

`prepare_voc.py --downloads <directory>` safely extracts the three original VOC
TAR releases into `data/raw/voc/`, verifies their original lists against pinned M
lists, and publishes the required 21,503 VOC image/XML pairs in `data/raw/ore/`.
Archive/member hashes and image/XML checks are recorded. This prepares the VOC
component only; it does not fabricate COCO XML or register an incomplete M benchmark.

`run_discovery_study.py` owns download → exact split preparation and image validation
→ discovery / discovery-no-message / discovery-scratch chains → paired final-epoch AP50.
Progress, subprocess logs and reports are stored under its output directory. Re-run
the same command to resume; no completed chain means no AP improvement conclusion.
Its single-GPU batch setting is recorded separately from the author data protocol.

`validate_discovery.py --output output/discovery/mechanisms.json mechanisms --device cuda`
checks controlled learning/state mechanics, not real-image AP. The `compare` subcommand
requires all seven completed training phases and actual saved training settings.
See [implementation, evidence and limitations](../../docs/owod-discovery-transition-design.md).

`local_transition_check.py` is the explicitly authorized local diagnostic: identical-capacity
graph/self-loop Task 1 comparison, followed by Task 2 transfer/scratch and separate author
balanced finetuning. Uses verified manifest hashes, official image identities/transforms and
the full test set. Short update budgets and disabled Task 2 risk suppression are recorded;
these are not official training schedules. Diagnostic states cannot initialize formal runs.
`--study optimization --steps 900 --task2-steps 300 --finetune-steps 120` evaluates
hierarchical versus focal-consistent independent scoring on identical weights, then
hard versus soft routing from the same fresh Task 1 parent with inheritance disabled.
Each evaluation forward produces both scorers plus a known-only diagnostic; the latter
is never presented as an OWOD result. Full author test coverage is required for every
reported comparison. Specify a fresh `--output` directory.
`local_discovery_check.py` records the earlier capacity-confounded Task 1 check; use the new
paired diagnostic for the current idea. Both preserve previous outputs.

`run_benchmark.py --method graph-adapter` runs persistent class adapters with
GNN risk gates, including calibration on completed tasks. See
[method configuration and limits](../../docs/graph-routed-adaptation.md).
