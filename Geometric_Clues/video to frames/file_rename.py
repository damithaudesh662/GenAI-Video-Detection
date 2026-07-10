

# save as rename_videos_keep_ext.py
from pathlib import Path

def rename_videos_sequential(folder="gen_ai", start=1):
    p = Path(folder)
    files = sorted([f for f in p.iterdir() if f.is_file()], key=lambda x: x.name)

    # temp-rename to avoid collisions
    temp = []
    for f in files:
        t = f.with_name(f.name + ".tmp_renaming")
        f.rename(t)
        temp.append(t)

    n = start
    for t in temp:
        ext = t.suffix.replace(".tmp_renaming", "") or ".mp4"  # fallback if none
        new_path = t.with_name(f"{n}{ext}")
        while new_path.exists():
            n += 1
            new_path = t.with_name(f"{n}{ext}")
        t.rename(new_path)
        n += 1

if __name__ == "__main__":
    rename_videos_sequential("gen_ai_new", 1)
