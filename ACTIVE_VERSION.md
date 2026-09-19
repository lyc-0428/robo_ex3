# Active version: real grid sorting final

This branch contains the final validated real-robot grid sorting program.

The implementation is located in `real/minified`. It preserves the successful
V10 recognition, grasp, transport, placement, and directional-search behavior.
Placement action counts are not treated as unique object identities, so a
recovered object does not cause the task to stop before the remaining objects
have been searched.

Run on the Jetson with:

```bash
cd /home/adam/Team21/yjh/real_grid_sorting_v10
bash ./run_grasp_v10.sh
```

The startup banner must show `BUILD: RESTORED-MOTION-COUNT-ONLY`.
