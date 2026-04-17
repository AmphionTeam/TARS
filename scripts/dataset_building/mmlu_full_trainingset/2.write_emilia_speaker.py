from datasets import load_dataset

path = "Emilia-YODAS/EN/*.tar"  # Same for Emilia-YODAS; just replace "Emilia/" with "Emilia-YODAS/"
dataset = load_dataset(
    "amphion/Emilia-Dataset", data_files={"en": path}, split="en", streaming=True
)


dataset = dataset.shuffle(seed=42 * 2)

dataset

dataset = dataset.filter(lambda info: info["duration"] <= 5, input_columns=["json"])

iterator = iter(dataset)

import torchaudio
import io
import os

outs = []
visited_speakers = set()

os.makedirs("/scratch/emilia_5s", exist_ok=True)

counter = 0
text = ""

for item in iterator:
    print(f"\r{counter}/20000, speakers: {len(visited_speakers)}, text: {text}", end="")
    if counter > 20000:
        break

    mp3 = item["mp3"]["bytes"]  # bytes
    text = item["json"]["text"]
    extra = item["json"]
    del extra["text"]

    if extra["speaker"] in visited_speakers:
        continue
    visited_speakers.add(extra["speaker"])

    # wav, sr = torchaudio.load(io.BytesIO(mp3))
    path = f"/scratch/emilia_5s/{extra['_id']}.wav"
    # torchaudio.save(path, wav, sample_rate=sr)
    with open(path, "wb") as f:
        f.write(mp3)

    outs.append({"key": extra["_id"], "text": text, "extra": extra, "wav_path": path})
    counter += 1

len(visited_speakers), len(outs)

from datasets import Dataset, Audio

outs_ds = Dataset.from_list(outs)
outs_ds = outs_ds.cast_column("wav_path", Audio())

outs_ds.save_to_disk(
    os.environ.get("EMILIA_DS_PATH", "/path/to/your/dataset/MMLU_FULL/emilia_5s_ds")
)
