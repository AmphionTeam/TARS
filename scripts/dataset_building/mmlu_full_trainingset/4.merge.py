# %%

from datasets import load_from_disk, Audio

ds = load_from_disk(
    os.environ.get(
        "TEXTONLY_DS_PATH",
        "/path/to/your/dataset/MMLU_FULL/full_train_hf_format.textonly.v0",
    )
)
ds = ds.shuffle(seed=42 * 3)

ds

# need to add 'audio', 'tts_wer', 'tts_retries'

# %%


base_path = os.environ.get(
    "WAV_BASE_PATH", "/path/to/your/dataset/MMLU_FULL/full_train_wav.v0"
)

ds = ds.map(lambda x: {"audio": f"{base_path}/wav/{x['key']}.wav"})  # 68739

ds[0]

# %%
import os

# os.system("cat /mnt/blob/v-chaorwang/MMLU_FULL/full_train_wav.v0/output_rank*.jsonl > /mnt/blob/v-chaorwang/MMLU_FULL/full_train_wav.v0/output_merged.jsonl")
# %%
with open(f"{base_path}/output_merged.jsonl", "r") as f:
    lines = f.readlines()
import json

print(len(lines), lines[0])

data_dict = {}

for line in lines:
    if line.strip():
        try:
            line = line.split("save_path")[-1]
            line = '{"save_path' + line
            data = json.loads(line)
            save_path = data.pop("save_path")
            if save_path:
                data_dict[save_path] = {
                    "tts_wer": data["wer"],
                    "tts_retries": data["retries"],
                    "tts_whisper": data["hypothesis_text"],
                    "tts_model": data.get("model", "openaudio-s1-mini"),
                }
        except:
            print("Error parsing line:", line)
            pass

len(data_dict)  # 67808
# %%

ds = ds.map(
    lambda x: data_dict.get(
        x,
        {"tts_wer": None, "tts_retries": None, "tts_whisper": None, "tts_model": None},
    ),
    input_columns=["audio"],
)

ds, ds[0]
# %%

ds = ds.filter(lambda x: x, input_columns=["tts_retries"])
ds  # 67808, removed: 68739-67808=931 (fail to generate...)
# %%
import numpy as np

tts_wers = ds["tts_wer"]

np.mean(tts_wers), len(tts_wers)  # 100%; 0.17617805646548979, 67808
# %%

_tmp = [w for w in tts_wers if w < 0.1]

np.mean(_tmp), len(_tmp)  # 10%; 0.03148743612443575, 48023
# %%

_tmp = [w for w in tts_wers if w < 0.08]

np.mean(_tmp), len(_tmp)  # 8%; 0.027908926551615712, 45218
# %%
# let's make it 10% threshold...

ds = ds.filter(lambda x: x < 0.1, input_columns=["tts_wer"])
ds  # 48023, removed: 67808-48023=19785
# %%

ds = ds.cast_column("audio", Audio())

# %%

# random get some to listen
from IPython.display import Audio as IPyAudio, display
import random

for _ in range(5):
    idx = random.randint(0, len(ds) - 1)
    print(
        f"Index: {ds[idx]['key']}, TTS WER: {ds[idx]['tts_wer']}, TTS retries: {ds[idx]['tts_retries']} TTS Model: {ds[idx]['tts_model']}"
    )
    print(f"prompt_for_tts: {ds[idx]['prompt_for_tts']}")
    print(f"tts_whisper: {ds[idx]['tts_whisper']}")
    display(IPyAudio(ds[idx]["audio"]["array"], rate=ds[idx]["audio"]["sampling_rate"]))
# %%

ds.save_to_disk(
    os.environ.get(
        "MERGED_DS_PATH",
        "/path/to/your/dataset/MMLU_FULL/full_train_hf_format.merged.v0",
    )
)
# %%
