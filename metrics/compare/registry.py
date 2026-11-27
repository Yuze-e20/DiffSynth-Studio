"""Shared paths, captions and video list for the full 70-video refined test set."""
import json
import os

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined"
ALL_DIR = os.path.join(ROOT, "all")
OURS_DAVIS = os.path.join(ROOT, "DAVIS2017_Diffutoon2")
OURS_PIX = os.path.join(ROOT, "pixabay100_nf328_ctx1g_s20_u05_t05")
CAP_DAVIS = "/mnt/nas3/sunyuzework/Diffutoon-2/data/DAVIS-MiraData_nf328_ctx1g_s20_u05_t05/DAVIS2017/_work"
CAP_PIX = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_Pixabay100"
METHODS = ["Source", "Ours", "EchoStyle", "PickStyle", "Kiwi-Edit", "AnyV2V", "Ref2VA", "Diffutoon", "Seedance25", "9blocks"]
GEN_METHODS = ["Ours", "EchoStyle", "PickStyle", "Kiwi-Edit", "AnyV2V", "Ref2VA", "Diffutoon", "Seedance25", "9blocks"]
STYLE_PREFIX = (
    "A Japanese anime-style cartoon frame with clean line work and flat cel-shaded colors."
)
EDIT_INSTRUCTION = (
    "Redraw this photo as a frame from a Japanese anime, with clean line art and flat cel-shaded colors."
)


def all_videos():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(ALL_DIR) if f.endswith(".mp4"))


def path_of(method, video):
    if method == "Source":
        return os.path.join(ALL_DIR, f"{video}.mp4")
    if method == "Ours":
        p1 = os.path.join(OURS_DAVIS, f"{video}_nf328_ctx1g_s20_u05_t05.mp4")
        if os.path.exists(p1):
            return p1
        for p in (
            os.path.join(OURS_PIX, f"{video}.mp4"),
            os.path.join(OURS_PIX, f"{video}_nf328_ctx1g_s20_u05_t05.mp4"),
        ):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Ours output missing for {video}")
    if method == "Ref2VA":
        p = os.path.join(ROOT, "Ref2VA", f"{video}_nf328_ctx1g.mp4")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    p = os.path.join(ROOT, method, f"{video}.mp4")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def global_caption(video):
    for p in (
        os.path.join(CAP_DAVIS, f"{video}_auto", "prompt_segments_w328_c1g.json"),
        os.path.join(CAP_PIX, video, f"{video}_auto", "prompt_segments_w328_c1g.json"),
    ):
        if os.path.exists(p):
            return json.load(open(p))["global_caption"].strip()
    raise FileNotFoundError(f"global caption missing for {video}")


def ur2_prompt(video):
    return f"{STYLE_PREFIX} {global_caption(video)}"


def edit_prompt(video):
    return f"{EDIT_INSTRUCTION} {global_caption(video)}"


def shard(items, index, n_shards):
    """Contiguous shard i of n (0-based)."""
    items = list(items)
    n = len(items)
    start = index * n // n_shards
    end = (index + 1) * n // n_shards
    return items[start:end]


def verify():
    vids = all_videos()
    assert len(vids) == 70, len(vids)
    for v in vids:
        global_caption(v)
        for m in METHODS:
            assert os.path.exists(path_of(m, v)), (m, v)
    print("registry ok:", len(vids), "videos x", len(METHODS), "methods")


if __name__ == "__main__":
    verify()
