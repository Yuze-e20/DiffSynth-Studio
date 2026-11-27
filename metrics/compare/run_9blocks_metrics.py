"""Full metrics for 9blocks on all 70."""
import os
METHOD = "9blocks"

def main():
    import metrics.compare.registry as reg
    if METHOD not in reg.METHODS:
        reg.METHODS = list(reg.METHODS) + [METHOD]
    if METHOD not in reg.GEN_METHODS:
        reg.GEN_METHODS = list(reg.GEN_METHODS) + [METHOD]
    from metrics.compare import run_full_metrics as rfm
    shard = os.environ.get("SHARD", "0/1")
    i, n = map(int, shard.split("/"))
    metric = os.environ["METRIC"]
    videos = reg.shard(reg.all_videos(), i, n)
    methods = [METHOD]
    print(f"9blocks metric={metric} shard={shard} nvid={len(videos)}", flush=True)
    if metric == "color":
        rfm.run_color(videos, methods)
    elif metric == "clip":
        rfm.run_clip(videos, methods)
    elif metric == "ur2":
        rfm.run_ur2(videos, methods)
    elif metric == "ur_edit":
        rfm.run_ur_edit(videos, methods)
    elif metric == "temporal":
        rfm.run_temporal(videos, methods, do_csd=True, do_smooth=True)
    else:
        raise SystemExit(metric)
    print("DONE", metric, shard, flush=True)

if __name__ == "__main__":
    main()
