# accelerate launch --num_processes 2 3.tts.py

from hashlib import md5
from datasets import load_from_disk, Audio
from gradio_client import Client, handle_file
import shutil
import os
import io
from tqdm import tqdm
from faster_whisper import WhisperModel
import jiwer
import json
import jsonlines
from accelerate import Accelerator


def wer(wav_path, reference_text):
    reference_text = reference_text.replace("\n", " ")

    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        word_timestamps=True,
        initial_prompt="Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. ",
    )

    result_text = []

    results = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "model_size": model_size,
        "segments": [],
    }

    for segment in segments:
        info = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
            "words": [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in segment.words
            ],
        }
        results["segments"].append(info)

        result_text.append(segment.text)

    # 示例文本
    hypothesis_text = "".join(result_text)

    # 定义预处理转换
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),  # 转换为小写
            jiwer.RemovePunctuation(),  # 移除标点符号
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),  # 去除多余的空白
        ]
    )

    processed_reference = transformation(reference_text)
    processed_hypothesis = transformation(hypothesis_text)

    if "option d" not in processed_hypothesis:
        print("Missing Option D.")
        return 1.0, hypothesis_text

    # 计算 WER
    wer_result = jiwer.wer(processed_reference, processed_hypothesis)
    print(f"WER: {wer_result * 100:.2f}%")

    return wer_result, hypothesis_text


def tts(text, save_path, reference_wav_path, reference_text):
    outs = []
    best_result = (None, 1)

    while best_result[1] > 0.05 and len(outs) < 10:  # wer more than 5%
        path = _tts(text, save_path, reference_wav_path, reference_text)
        wer_result, hypothesis_text = wer(path, text)

        outs.append((path, wer_result))
        best_result = sorted(outs, key=lambda x: x[1])[0]  # update best_result

    path = best_result[0]
    wer_result = best_result[1]

    shutil.move(path, save_path)

    r = {
        "save_path": save_path,
        "model": "cosyvoice",  # newly added
        "text": text,
        "wer": wer_result,
        "retries": len(outs),
        "hypothesis_text": hypothesis_text,
    }

    with open(save_path.replace(".wav", ".json"), "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    return r


def _tts(text, save_path, reference_wav_path, reference_text):
    return _tts_cosyvoice(text, save_path, reference_wav_path, reference_text)


def _tts_cosyvoice(text, save_path, reference_wav_path, reference_text, retries=3):
    if retries == 0:
        return None

    try:
        result = client.predict(
            text=text,
            reference_text=reference_text,
            reference_audio_path=handle_file(reference_wav_path),
            api_name="/generate_audio",
        )
        path, err = result
    except Exception as e:
        err = str(e)
        print(e)

    if err:
        return _tts_cosyvoice(
            text, save_path, reference_wav_path, reference_text, retries - 1
        )

    return path


def _tts_opentts(text, save_path, reference_wav_path, reference_text, retries=3):

    if retries == 0:
        return None

    try:
        result = client.predict(
            text=text,
            reference_id=str(md5(reference_text.encode("utf-8")).hexdigest()),
            reference_audio=handle_file(reference_wav_path),
            reference_text=reference_text,
            max_new_tokens=0,
            chunk_length=0,
            top_p=0.9,
            repetition_penalty=1.0,
            temperature=0.9,
            seed=0,
            use_memory_cache="on",
            api_name="/partial",
        )
        path, err = result
    except Exception as e:
        err = str(e)
        print(e)

    if err:
        return _tts_opentts(
            text, save_path, reference_wav_path, reference_text, retries - 1
        )

    return path


if __name__ == "__main__":

    accelerator = Accelerator()

    model_size = "medium"
    with accelerator.local_main_process_first():
        model = WhisperModel(
            model_size,
            device="cuda",
            device_index=accelerator.local_process_index,
            compute_type="float16",
        )

        ds = load_from_disk(
            os.environ.get(
                "TEXTONLY_DS_PATH",
                "/path/to/your/dataset/MMLU_FULL/full_train_hf_format.textonly.v0",
            )
        )
        ds = ds.shuffle(seed=42 * 3)

        emilia_ds = load_from_disk(
            os.environ.get(
                "EMILIA_DS_PATH", "/path/to/your/dataset/MMLU_FULL/emilia_5s_ds"
            )
        )
        emilia_ds = emilia_ds.shuffle(seed=42)

        emilia_ds = emilia_ds.cast_column("wav_path", Audio(decode=False))

        base_path = os.environ.get(
            "WAV_BASE_PATH", "/path/to/your/dataset/MMLU_FULL/full_train_wav.v0"
        )

        os.makedirs(base_path, exist_ok=True)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    print(f"{rank = }, {world_size = }")

    ds = ds.shard(num_shards=world_size, index=rank)

    port = 7860 + accelerator.local_process_index
    client = Client(f"http://127.0.0.1:{port}/")

    pb = tqdm(ds, total=len(ds), desc="Processing")

    os.makedirs(f"{base_path}/wav", exist_ok=True)
    os.makedirs("/scratch/emilia_5s", exist_ok=True)

    for item in pb:
        text = item["prompt_for_tts"]
        save_path = f"{base_path}/wav/{item['key']}.wav"

        idx = item["question_id"] % 20000

        assigned_speaker = emilia_ds[idx]
        reference_wav_path = assigned_speaker["wav_path"]["path"]
        reference_wav_path = os.path.join("/scratch/emilia_5s", reference_wav_path)

        data = assigned_speaker["wav_path"]["bytes"]
        with open(reference_wav_path, "wb") as f:
            f.write(data)

        reference_text = assigned_speaker["text"]

        if os.path.exists(save_path):
            print(f"File {save_path} already exists")
        else:
            r = tts(text, save_path, reference_wav_path, reference_text)

            with jsonlines.open(
                f"{base_path}/output_rank{rank+100}.jsonl", mode="a"
            ) as writer:
                writer.write(r)

        pb.set_description(f"{item['key']}, text: {text[:50]}...")

    pb.close()
