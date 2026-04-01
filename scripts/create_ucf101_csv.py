"""Generate space-delimited train/test CSVs for UCF101 (split 1).

Output format (matches SSv2 CSV convention):
  /abs/path/to/video.avi 0
  /abs/path/to/video2.avi 3
"""

import argparse
import os


def load_class_index(split_dir):
    """Read classInd.txt -> {class_name: 0-indexed label}."""
    path = os.path.join(split_dir, "classInd.txt")
    mapping = {}
    with open(path) as f:
        for line in f:
            idx, name = line.strip().split()
            mapping[name] = int(idx) - 1  # convert 1-indexed to 0-indexed
    return mapping


def make_csv(split_file, video_dir, class_map, output_path):
    """Read a UCF101 split file and write a space-delimited CSV."""
    with open(split_file) as f_in, open(output_path, "w") as f_out:
        count = 0
        for line in f_in:
            # trainlist has "class/video.avi label", testlist has just "class/video.avi"
            parts = line.strip().split()
            rel_path = parts[0]
            class_name = rel_path.split("/")[0]
            label = class_map[class_name]
            abs_path = os.path.join(video_dir, rel_path)
            if not os.path.exists(abs_path):
                print(f"WARNING: missing {abs_path}")
                continue
            f_out.write(f"{abs_path} {label}\n")
            count += 1
    print(f"Wrote {count} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Path to UCF-101/ directory")
    parser.add_argument("--split_dir", required=True, help="Path to ucfTrainTestlist/ directory")
    parser.add_argument("--output_dir", required=True, help="Where to write train.csv and test.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_map = load_class_index(args.split_dir)
    print(f"Loaded {len(class_map)} classes")

    make_csv(
        os.path.join(args.split_dir, "trainlist01.txt"),
        args.video_dir, class_map,
        os.path.join(args.output_dir, "train.csv"),
    )
    make_csv(
        os.path.join(args.split_dir, "testlist01.txt"),
        args.video_dir, class_map,
        os.path.join(args.output_dir, "test.csv"),
    )


if __name__ == "__main__":
    main()
