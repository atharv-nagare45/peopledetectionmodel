from pathlib import Path

dataset = Path(".")

for split in ["train", "valid", "test"]:
    labels_folder = dataset / split / "labels"

    if not labels_folder.exists():
        continue

    for txt_file in labels_folder.glob("*.txt"):
        lines = txt_file.read_text().splitlines()

        new_lines = []

        for line in lines:
            parts = line.split()

            if parts:
                # Class 0 (people) and Class 1 (person)
                # both become Class 0 (person)
                parts[0] = "0"
                new_lines.append(" ".join(parts))

        txt_file.write_text("\n".join(new_lines))

print("Done! Both classes are now person.")