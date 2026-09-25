# VSLAM passing baseline — 2026-09-25

This directory and the annotated `vslam-pass-20260925` Git tag preserve the
passing code, exact configuration, saved RTAB-Map database and both phase reports.
The map is losslessly gzip-compressed; it is included in Git, not just left in an
ignored runtime directory. Do not use this copy as a writable mapping database.

Uncompressed database SHA-256:
`4d9a47a4a9f584350202fa89b1200f91b1a65bf3869afa522079b368fe082872`.

Baseline result: mapping 14/14, navigation 4/4, zero contact episodes,
return error 0.133 m, yaw error 1.32 degrees, recovery after uncovering 0.140 s.
This is a passing sample, not a statistical reliability claim. Original report
paths describe the workstation on which it was generated; the preserved JSON
files and compressed database here are the portable snapshot.

To inspect or reproduce without overwriting ongoing work, create a separate
checkout from the tag and decompress the map into a new runtime location:

```bash
git worktree add --detach ../ysgo2-vslam-baseline vslam-pass-20260925
cd ../ysgo2-vslam-baseline
mkdir -p maps
gzip -dc baselines/vslam-pass-20260925/map.rtabmap.db.gz > maps/baseline.rtabmap.db
sha256sum maps/baseline.rtabmap.db
./start_go2_vslam.sh --localization --no-rviz \
  --database="$PWD/maps/baseline.rtabmap.db" --headless \
  --vslam-benchmark-phase localization \
  --vslam-benchmark-config "$PWD/baselines/vslam-pass-20260925/benchmark.yaml" \
  --vslam-benchmark-output "$PWD/reports/baseline-replay/localization.json"
```

The ROS, MuJoCo and Python dependencies described in the project documentation
are still required. Scene geometry and policy weights are versioned elsewhere
in this same repository.
