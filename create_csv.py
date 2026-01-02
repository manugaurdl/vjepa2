import json
import os
from pathlib import Path

##args
split = "validation"

data_dir = Path("/nas/manu/ssv2")
train_json = os.path.join(data_dir, f"labels/{split}.json")
labels_json = os.path.join(data_dir, "labels/labels.json")
video_dir = os.path.join(data_dir, "20bn-something-something-v2")  # directory containing videos
out_csv = os.path.join(data_dir, f"data/{split}.csv")
# os.makedirs(os.path.dirname(os.path.join(data_dir, "data")), exist_ok=True)

train = json.loads(open(train_json).read())
label_map = json.loads(open(labels_json).read())  # template -> "idx"
label_map = {k.lower(): v for k,v in label_map.items()}
missing = 0
with open(out_csv, "w") as f:
    for ex in train:
        vid = ex["id"]
        template = ex["template"].lower().replace("[", "").replace("]", "")
        try:
            y = int(label_map[template])
        except:
            breakpoint

        # try common extensions
        file_path = os.path.join(video_dir, f"{vid}.webm")
        # if not path.exists():
        #     path = video_dir / f"{vid}.mp4"
        if not os.path.isfile(file_path):
            missing += 1
            print(missing)
            continue

        # space-delimited, no header
        f.write(f"{Path(file_path).resolve()} {y}\n")

print("wrote", out_csv, "missing_videos", missing, "num_classes", len(label_map))