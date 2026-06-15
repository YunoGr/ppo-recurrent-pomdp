from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

video_dir = ROOT / "results" / "videos"
gif_dir = ROOT / "results" / "gifs"
fig_dir = ROOT / "report" / "figures"

gif_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)


VIDEOS = [
    {
        "file": "random_cartpole.mp4",
        "label": "Random agent",
        "caption": "Fails quickly",
    },
    {
        "file": "ppo_mlp_mdp_seed1.mp4",
        "label": "PPO-MLP / MDP",
        "caption": "Full observation",
    },
    {
        "file": "ppo_mlp_pomdp_seed1.mp4",
        "label": "PPO-MLP / POMDP",
        "caption": "Velocity masked",
    },
    {
        "file": "ppo_framestack_pomdp_seed3.mp4",
        "label": "PPO-FrameStack / POMDP",
        "caption": "Explicit memory",
    },
    {
        "file": "sb3_recurrentppo_pomdp_seed2.mp4",
        "label": "RecurrentPPO-LSTM / POMDP",
        "caption": "Recurrent memory",
    },
]


def resize_frame(frame, width=480):
    img = Image.fromarray(frame)
    w, h = img.size
    new_h = int(h * width / w)
    return img.resize((width, new_h))


def make_gif(video_path, out_path, max_frames=120, fps=10):
    reader = imageio.get_reader(video_path)
    frames = []

    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        frames.append(resize_frame(frame, width=480))

    reader.close()

    if not frames:
        print(f"[WARN] No frames found for {video_path}")
        return

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )

    print(f"GIF sauvegardé : {out_path}")


def extract_representative_frame(video_path, mode="middle"):
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()

    if not frames:
        raise ValueError(f"No frames found in {video_path}")

    if mode == "last":
        idx = len(frames) - 1
    elif mode == "early":
        idx = min(10, len(frames) - 1)
    else:
        idx = len(frames) // 2

    return frames[idx]


def make_snapshot_figure():
    fig, axes = plt.subplots(1, len(VIDEOS), figsize=(16, 3.5))

    for ax, item in zip(axes, VIDEOS):
        video_path = video_dir / item["file"]

        mode = "last" if "random" in item["file"] or "pomdp_seed1" in item["file"] else "middle"
        frame = extract_representative_frame(video_path, mode=mode)

        ax.imshow(frame)
        ax.set_title(item["label"], fontsize=10)
        ax.set_xlabel(item["caption"], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    out_path = fig_dir / "cartpole_agent_snapshots.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Figure snapshots sauvegardée : {out_path}")


def main():
    print("=" * 80)
    print("PHASE G — GIFs et captures visuelles")
    print("=" * 80)

    for item in VIDEOS:
        video_path = video_dir / item["file"]
        gif_path = gif_dir / item["file"].replace(".mp4", ".gif")

        if not video_path.exists():
            print(f"[WARN] Vidéo absente : {video_path}")
            continue

        make_gif(video_path, gif_path)

    make_snapshot_figure()

    print("=" * 80)
    print(f"GIFs disponibles dans : {gif_dir}")
    print(f"Figure disponible dans : {fig_dir / 'cartpole_agent_snapshots.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()