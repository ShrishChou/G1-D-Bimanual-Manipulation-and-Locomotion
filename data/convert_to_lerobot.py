"""Convert per-frame teleop recordings -> GR00T-ready LeRobot v2.0 dataset.

Expects a source layout of `episode_XXXX/` folders, each containing `data.json` (per-frame states/actions)
and `colors/{frame:06d}_color_{0,1,2}.jpg` (head + two wrist cameras).

  - 3 cameras: color_0=head, color_1=left wrist, color_2=right wrist -> 3 MP4 video streams.
  - 28-dim state/action: left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7).
  - Re-indexes (possibly gapped) source episodes to contiguous 0..N-1 as LeRobot requires.
  - --delta stores residual-to-current actions (action = target_qpos - measured_state); state stays absolute.
    Match this at deploy time (target = measured_state + policy_output).
  - Writes meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}. Normalization stats are computed
    by GR00T on first load, so delta datasets get their own stats automatically.

  python data/convert_to_lerobot.py --src /path/to/raw_dataset --out /path/to/lerobot_dataset [--delta]
  # or select episodes across datasets by a split manifest:
  python data/convert_to_lerobot.py --manifest split.json --key train --out /path/to/lerobot_dataset
"""
import argparse, glob, json, os
import cv2, numpy as np, pandas as pd

CAMS = [("color_0", "observation.images.head"),
        ("color_1", "observation.images.left_wrist"),
        ("color_2", "observation.images.right_wrist")]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
HAND = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
STATE_NAMES = ([f"left_{j}" for j in ARM] + [f"right_{j}" for j in ARM] +
               [f"left_hand_{j}" for j in HAND] + [f"right_hand_{j}" for j in HAND])

p = argparse.ArgumentParser()
p.add_argument("--src", help="single dataset root (globs episode_*)")
p.add_argument("--manifest", help="split json: pick episodes across datasets by --key")
p.add_argument("--key", help="which manifest list to convert")
p.add_argument("--out", required=True)
p.add_argument("--delta", action="store_true", help="store residual-to-current actions")
p.add_argument("--link-videos-from", default="", help="symlink videos from a twin dataset (same episode order)")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--jpeg-fps", type=float, default=0.0)
args = p.parse_args()


def vec(dp, kind):
    d = dp[kind]
    return (list(d["left_arm"]["qpos"]) + list(d["right_arm"]["qpos"]) +
            list(d["left_ee"]["qpos"]) + list(d["right_ee"]["qpos"]))


if args.manifest:
    man = json.load(open(args.manifest))
    src_eps = [f"{man['base']}/{eid}" for eid in man[args.key]]
    src_eps = [e for e in src_eps if os.path.exists(e + "/data.json")]
elif args.src:
    src_eps = sorted(d for d in glob.glob(f"{args.src}/episode_*") if os.path.exists(d + "/data.json"))
else:
    raise SystemExit("provide --src OR --manifest/--key")
if args.limit:
    src_eps = src_eps[:args.limit]
if not src_eps:
    raise SystemExit("no episodes with data.json selected")

os.makedirs(f"{args.out}/data/chunk-000", exist_ok=True)
os.makedirs(f"{args.out}/meta", exist_ok=True)
for _, vk in CAMS:
    os.makedirs(f"{args.out}/videos/chunk-000/{vk}", exist_ok=True)

episodes_meta, total_frames, task, fps, H, W = [], 0, None, 30.0, 480, 640
global_index = 0
for new_idx, ep in enumerate(src_eps):
    data = json.load(open(ep + "/data.json"))
    frames = data["data"]
    task = data["text"]["goal"]
    fps = float(args.jpeg_fps or data["info"]["image"]["fps"])
    dt = 1.0 / fps
    rows = []
    for k, dp in enumerate(frames):
        s, a = vec(dp, "states"), vec(dp, "actions")
        assert len(s) == 28 and len(a) == 28, f"{ep} frame {k}: dim {len(s)}/{len(a)} != 28"
        if args.delta:
            a = [ai - si for ai, si in zip(a, s)]
        rows.append({"observation.state": s, "action": a, "timestamp": k * dt,
                     "frame_index": k, "episode_index": new_idx, "index": global_index + k, "task_index": 0})
    pd.DataFrame(rows).to_parquet(f"{args.out}/data/chunk-000/episode_{new_idx:06d}.parquet")
    global_index += len(frames)

    if args.link_videos_from:
        for _, vk in CAMS:
            src_mp4 = os.path.abspath(f"{args.link_videos_from}/videos/chunk-000/{vk}/episode_{new_idx:06d}.mp4")
            dst_mp4 = f"{args.out}/videos/chunk-000/{vk}/episode_{new_idx:06d}.mp4"
            if not os.path.exists(dst_mp4):
                os.symlink(src_mp4, dst_mp4)
    else:
        for cam_key, vk in CAMS:
            vw = cv2.VideoWriter(f"{args.out}/videos/chunk-000/{vk}/episode_{new_idx:06d}.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            for fpath in sorted(glob.glob(f"{ep}/colors/*_{cam_key}.jpg")):
                img = cv2.imread(fpath)
                if img is None:
                    continue
                vw.write(cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img)
            vw.release()
    episodes_meta.append({"episode_index": new_idx, "tasks": [task], "length": len(frames)})
    total_frames += len(frames)
    print(f"[convert] {os.path.basename(ep)} -> episode_{new_idx:06d}  ({len(frames)} frames)")

with open(f"{args.out}/meta/episodes.jsonl", "w") as f:
    for e in episodes_meta:
        f.write(json.dumps(e) + "\n")
with open(f"{args.out}/meta/tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
modality = {
    "state":  {"left_arm": {"start": 0, "end": 7}, "right_arm": {"start": 7, "end": 14},
               "left_hand": {"start": 14, "end": 21}, "right_hand": {"start": 21, "end": 28}},
    "action": {"left_arm": {"start": 0, "end": 7}, "right_arm": {"start": 7, "end": 14},
               "left_hand": {"start": 14, "end": 21}, "right_hand": {"start": 21, "end": 28}},
    "video": {"head": {"original_key": "observation.images.head"},
              "left_wrist": {"original_key": "observation.images.left_wrist"},
              "right_wrist": {"original_key": "observation.images.right_wrist"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
json.dump(modality, open(f"{args.out}/meta/modality.json", "w"), indent=4)


def vid_feature():
    return {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"],
            "info": {"video.height": H, "video.width": W, "video.codec": "mp4v", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False}}


features = {vk: vid_feature() for _, vk in CAMS}
features["observation.state"] = {"dtype": "float64", "shape": [28], "names": STATE_NAMES}
features["action"] = {"dtype": "float64", "shape": [28], "names": STATE_NAMES}
for col, dt_ in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
                 ("index", "int64"), ("task_index", "int64")):
    features[col] = {"dtype": dt_, "shape": [1], "names": None}
info = {"codebase_version": "v2.0", "robot_type": "Unitree_G1_D", "total_episodes": len(src_eps),
        "total_frames": total_frames, "total_tasks": 1, "total_videos": len(src_eps) * len(CAMS),
        "total_chunks": 1, "chunks_size": len(src_eps), "fps": fps, "splits": {"train": f"0:{len(src_eps)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features, "discarded_episode_indices": []}
json.dump(info, open(f"{args.out}/meta/info.json", "w"), indent=4)
json.dump({"delta": bool(args.delta)}, open(f"{args.out}/meta/action_space.json", "w"), indent=2)
print(f"\n[convert] DONE: {len(src_eps)} episodes, {total_frames} frames, "
      f"action={'DELTA' if args.delta else 'ABSOLUTE'} -> {args.out}")
