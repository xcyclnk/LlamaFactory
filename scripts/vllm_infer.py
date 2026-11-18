# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import hashlib
import json
import math
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import av
import fire
from datasets import load_dataset
from eval_bleu_rouge import compute_metrics
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def _need_video_kwargs(template):
    NEEDED_TEMPLATE = ["qwen3_vl", "glm4v"]
    if any(t in template for t in NEEDED_TEMPLATE):
        return True

    return False


def _extract_mcq_answer(text: str, label: str) -> str | None:
    r"""Extract a multiple-choice letter when the reference label is a letter."""
    normalized_label = str(label).strip().upper()
    if not re.fullmatch(r"[A-Z]", normalized_label):
        return None

    # CG-Bench currently uses choices A-H. Restricting the alphabet prevents
    # letters in ordinary prose from being mistaken for an answer.
    allowed = "ABCDEFGH"
    tail = str(text).rsplit("</think>", 1)[-1].strip().upper()
    for pattern in (
        rf"<ANSWER>\s*([{allowed}])\s*</ANSWER>",
        rf"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\[(]?([{allowed}])",
        rf"^\s*[\[(]?([{allowed}])[\])\].,:]?\s*$",
    ):
        match = re.search(pattern, tail)
        if match:
            return match.group(1)

    matches = re.findall(rf"(?<![A-Z])([{allowed}])(?![A-Z])", tail)
    return matches[-1] if matches else None


def _video_uuid(
    video_path: str,
    video_fps: float,
    video_maxlen: int,
    image_min_pixels: int,
    image_max_pixels: int,
) -> str:
    r"""Build a stable cache ID that changes with the file or sampling setup."""
    path = Path(video_path).resolve()
    stat = path.stat()
    payload = "|".join(
        (
            str(path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(video_fps),
            str(video_maxlen),
            str(image_min_pixels),
            str(image_max_pixels),
        )
    )
    return f"video-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _video_sort_key(video: object) -> str:
    if isinstance(video, list) and video:
        first = video[0]
        if isinstance(first, (str, os.PathLike)):
            return str(Path(first).resolve())

    return ""


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: int | None = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    matrix_save_name: str = None,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: str | None = None,
    enable_thinking: bool = True,
    seed: int | None = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 8,
    group_by_video: bool = False,
    video_cache_size: int = 2,
    use_video_uuid: bool = True,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if video_cache_size < 1:
        raise ValueError("video_cache_size must be positive.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    model_preparation_start_time = time.time()
    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]
    need_video_kwargs = _need_video_kwargs(template)

    processing_indices = list(range(len(train_dataset)))
    if group_by_video:
        processing_indices.sort(key=lambda index: (_video_sort_key(train_dataset[index]["videos"]), index))

    @lru_cache(maxsize=video_cache_size)
    def prepare_video(video_path: str):
        regularized = template_obj.mm_plugin._regularize_videos(
            [video_path],
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
            video_fps=video_fps,
            video_maxlen=video_maxlen,
        )["videos"]

        video_metadata_kwargs = None
        video_metadata = None
        if need_video_kwargs:
            with av.open(video_path, "r") as container:
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sampling_indices = template_obj.mm_plugin._get_video_sample_indices(
                    video_stream,
                    video_fps,
                    video_maxlen,
                )
                total_frames = video_stream.frames

            video_metadata_kwargs = {
                "fps": getattr(tokenizer_module["processor"], "video_fps", 24.0),
                "do_sample_frames": False,
                "total_num_frames": total_frames,
            }
            video_metadata = {
                "fps": video_fps,
                "frames_indices": sampling_indices,
                "total_num_frames": total_frames,
                "video_backend": "opencv",
            }

        return regularized, video_metadata_kwargs, video_metadata

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Index results by original dataset position so grouping does not break
    # compatibility with scorers that zip predictions with a manifest.
    all_records = [None] * len(train_dataset)

    model_predict_start_time = time.time()
    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = save_path.with_suffix(save_path.suffix + ".partial")
    progress_path = save_path.with_suffix(save_path.suffix + ".progress.json")
    partial_file = partial_path.open("w", encoding="utf-8")
    parsed_count = 0
    correct_count = 0
    mcq_count = 0

    # Add batch process to avoid the issue of too many files opened
    batch_starts = range(0, len(processing_indices), batch_size)
    for batch_number, i in enumerate(tqdm(batch_starts, desc="Processing batched inference"), start=1):
        batch_indices = processing_indices[i : min(i + batch_size, len(processing_indices))]
        vllm_inputs, prompts, labels, media_infos = [], [], [], []
        batch = train_dataset[batch_indices]

        for j in range(len(batch["input_ids"])):
            multi_modal_data = {}
            video_metadata_kwargs = None
            media_info = {"video_path": None, "video_uuid": None}

            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data["image"] = template_obj.mm_plugin._regularize_images(
                    image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                )["images"]

            if batch["videos"][j] is not None:
                video = batch["videos"][j]
                video_input = video[0]
                if isinstance(video_input, (str, os.PathLike)):
                    video_path = str(Path(video_input).resolve())
                    regularized, cached_video_kwargs, cached_video_metadata = prepare_video(video_path)
                    multi_modal_data["video"] = regularized
                    video_metadata_kwargs = cached_video_kwargs
                    video_metadata = cached_video_metadata
                    media_info["video_path"] = video_path
                    if use_video_uuid:
                        media_info["video_uuid"] = _video_uuid(
                            video_path,
                            video_fps,
                            video_maxlen,
                            image_min_pixels,
                            image_max_pixels,
                        )
                else:
                    multi_modal_data["video"] = template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                if need_video_kwargs and not isinstance(video_input, (str, os.PathLike)):
                    # Already sampled frame list.
                    total_frames = len(video_input)
                    video_metadata_kwargs = {
                        "do_sample_frames": False,
                        "total_num_frames": total_frames,
                    }
                    video_metadata = {
                        "fps": video_fps,
                        "frames_indices": list(range(total_frames)),
                        "total_num_frames": total_frames,
                        "video_backend": "opencv",
                    }

                    multi_modal_data["video"] = (
                        multi_modal_data["video"],
                        video_metadata,
                    )

                if need_video_kwargs and isinstance(video_input, (str, os.PathLike)):
                    multi_modal_data["video"] = (
                        multi_modal_data["video"],
                        video_metadata,
                    )

            if batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data["audio"] = zip(audio_data["audios"], audio_data["sampling_rates"])

            vllm_input_data = {"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data or None}
            if video_metadata_kwargs is not None:
                vllm_input_data["mm_processor_kwargs"] = video_metadata_kwargs
            if media_info["video_uuid"] is not None:
                vllm_input_data["multi_modal_uuids"] = {"video": [media_info["video_uuid"]]}

            vllm_inputs.append(vllm_input_data)
            prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )
            media_infos.append(media_info)

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        for original_index, prompt, pred, label, media_info in zip(
            batch_indices, prompts, preds, labels, media_infos
        ):
            parsed_answer = _extract_mcq_answer(pred, label)
            normalized_label = str(label).strip().upper()
            is_mcq = re.fullmatch(r"[A-Z]", normalized_label) is not None
            correct = parsed_answer == normalized_label if is_mcq else None
            mcq_count += int(is_mcq)
            parsed_count += int(is_mcq and parsed_answer is not None)
            correct_count += int(correct is True)
            record = {
                "sample_index": original_index,
                "prompt": prompt,
                "predict": pred,
                "label": label,
                "parsed_answer": parsed_answer,
                "correct": correct,
                **media_info,
            }
            all_records[original_index] = record
            partial_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        partial_file.flush()
        cache_info = prepare_video.cache_info()
        completed = min(i + len(batch_indices), len(processing_indices))
        elapsed = time.time() - model_predict_start_time
        running_accuracy = correct_count / mcq_count if mcq_count else None
        progress = {
            "completed": completed,
            "total": len(processing_indices),
            "batch": batch_number,
            "elapsed_seconds": elapsed,
            "samples_per_second": completed / elapsed if elapsed > 0 else 0.0,
            "video_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "maxsize": cache_info.maxsize,
                "currsize": cache_info.currsize,
            },
            "mcq": {
                "parsed": parsed_count,
                "correct": correct_count,
                "total": mcq_count,
                "accuracy": running_accuracy,
            },
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
        accuracy_text = f"{running_accuracy:.4%}" if running_accuracy is not None else "n/a"
        print(
            f"[observable] completed={completed}/{len(processing_indices)} "
            f"speed={progress['samples_per_second']:.4f} sample/s "
            f"video_cache={cache_info.hits} hit/{cache_info.misses} miss "
            f"mcq_parsed={parsed_count}/{mcq_count} running_acc={accuracy_text}"
        )
        gc.collect()

    partial_file.close()
    model_predict_end_time = time.time()
    missing = [index for index, record in enumerate(all_records) if record is None]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions; first missing sample index: {missing[0]}")

    # Restore original dataset order for compatibility with external manifests.
    with save_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    predict_runtime = model_predict_end_time - model_predict_start_time
    final_cache_info = prepare_video.cache_info()
    cache_requests = final_cache_info.hits + final_cache_info.misses
    summary = {
        "model_name_or_path": model_args.model_name_or_path,
        "dataset": dataset,
        "template": template,
        "samples": len(all_records),
        "runtime_seconds": predict_runtime,
        "samples_per_second": len(all_records) / predict_runtime if predict_runtime > 0 else 0.0,
        "batch_size": batch_size,
        "cutoff_len": cutoff_len,
        "max_new_tokens": max_new_tokens,
        "video_fps": video_fps,
        "video_maxlen": video_maxlen,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "group_by_video": group_by_video,
        "video_cache": {
            "hits": final_cache_info.hits,
            "misses": final_cache_info.misses,
            "maxsize": final_cache_info.maxsize,
            "currsize": final_cache_info.currsize,
            "hit_rate": final_cache_info.hits / cache_requests if cache_requests else None,
        },
        "mcq": {
            "parsed": parsed_count,
            "parse_failures": mcq_count - parsed_count,
            "correct": correct_count,
            "total": mcq_count,
            "accuracy": correct_count / mcq_count if mcq_count else None,
        },
        "predictions": str(save_path),
        "partial_predictions": str(partial_path),
        "progress": str(progress_path),
    }
    summary_path = save_path.with_suffix(save_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("*" * 70)
    print(f"{len(all_records)} total generated results have been saved at {save_name}.")
    print(f"Observable summary has been saved at {summary_path}.")
    if mcq_count:
        print(
            f"MCQ parsed={parsed_count}/{mcq_count}, correct={correct_count}/{mcq_count}, "
            f"accuracy={correct_count / mcq_count:.4%}."
        )
    print("*" * 70)

    # Write all matrix results when matrix_save_name is not None,
    # The result matrix is referencing src.llamafactory.train.sft.workflow.run_sft # 127~132
    # trainer.save_metrics("predict", predict_results.metrics)
    #
    #   {
    #        "predict_bleu-4": 4.349975,
    #        "predict_model_preparation_time": 0.0128,
    #        "predict_rouge-1": 21.873359375,
    #        "predict_rouge-2": 4.144340625,
    #        "predict_rouge-l": 10.83949375,
    #        "predict_runtime": 131.664,
    #        "predict_samples_per_second": 0.076,
    #        "predict_steps_per_second": 0.008
    #    }
    #
    if matrix_save_name is not None:
        predict_time = model_predict_end_time - model_predict_start_time
        preparation_time = model_predict_start_time - model_preparation_start_time

        start_time = time.time()
        dataset = load_dataset("json", data_files=save_name, split="train")
        dataset = dataset.map(compute_metrics, num_proc=8, remove_columns=dataset.column_names)
        score_dict = dataset.to_dict()

        average_score = {}
        for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
            score = sum(scores) / len(scores) if scores else 0.0
            print(f"predict_{task}: {score:.4f}")
            average_score["predict_" + task] = score

        average_score["predict_model_preparation_time"] = preparation_time
        average_score["predict_runtime"] = predict_time
        num_steps = math.ceil(len(train_dataset) / batch_size)
        average_score["predict_samples_per_second"] = len(dataset) / predict_time if predict_time > 0 else 0.0
        average_score["predict_steps_per_second"] = num_steps / predict_time if predict_time > 0 else 0.0

        with open(matrix_save_name, "w", encoding="utf-8") as f:
            json.dump(average_score, f, indent=4)

        print("*" * 70)
        print(f"\nDone in {time.time() - start_time:.3f}s.\nScore file saved to {matrix_save_name}.")
        print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)
