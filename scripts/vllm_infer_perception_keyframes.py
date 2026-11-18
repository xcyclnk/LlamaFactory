#!/usr/bin/env python3
"""Evaluate compact CG-Bench questions from dynamically resolved event keyframes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path.cwd())
    parser.add_argument("--template", default="qwen3_vl_nothink")
    parser.add_argument("--cutoff-len", type=int, default=32768)
    parser.add_argument("--image-min-pixels", type=int, default=4096)
    parser.add_argument("--image-max-pixels", type=int, default=65536)
    parser.add_argument(
        "--max-events",
        type=int,
        default=128,
        help="Fail if a video exceeds this event count; 0 disables the guard.",
    )
    parser.add_argument("--request-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vllm-config",
        default='{"gpu_memory_utilization":0.90,"max_num_seqs":1,'
        '"enable_prefix_caching":true,"mm_processor_cache_gb":4}',
    )
    parser.add_argument("--save-name", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not records:
        raise ValueError(f"No questions found in {path}")

    required = {"row_index", "qid", "video_uid", "question", "choices", "answer"}
    for position, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"Question row {position} is missing: {', '.join(missing)}")
    row_indices = [int(record["row_index"]) for record in records]
    if sorted(row_indices) != list(range(len(records))):
        raise ValueError("row_index values must be unique and contiguous from zero")
    return records


def parse_mmss(value: Any) -> float:
    match = re.fullmatch(r"(\d+):([0-5]\d(?:\.\d+)?)", str(value).strip())
    if not match:
        raise ValueError(f"Invalid MM:SS timestamp: {value!r}")
    return int(match.group(1)) * 60.0 + float(match.group(2))


def event_interval_seconds(event: dict[str, Any]) -> tuple[float, float]:
    if "time_min" in event:
        values = event["time_min"]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"Invalid time_min: {values!r}")
        start, end = (float(value) * 60.0 for value in values)
    elif "time_mmss" in event:
        values = event["time_mmss"]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"Invalid time_mmss: {values!r}")
        start, end = (parse_mmss(value) for value in values)
    else:
        raise ValueError("Event contains neither time_min nor time_mmss")
    if not all(math.isfinite(value) and value >= 0 for value in (start, end)) or end < start:
        raise ValueError(f"Invalid event interval: {(start, end)!r}")
    return start, end


def format_timestamp(seconds: float) -> str:
    centiseconds = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def resolve_keyframe(frame_dir: Any, path_root: Path, memory_path: Path) -> Path:
    raw_keyframe = Path(str(frame_dir)) / "k.jpg"
    candidates = [raw_keyframe]
    if not raw_keyframe.is_absolute():
        candidates.extend(
            (
                path_root / raw_keyframe,
                memory_path.parent / raw_keyframe,
                memory_path.parent.parent / raw_keyframe,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing keyframe; tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


def load_event_bank(
    video_uid: str,
    memory_dir: Path,
    path_root: Path,
    max_events: int,
) -> tuple[Path, list[dict[str, Any]]]:
    memory_path = memory_dir / f"{video_uid}.json"
    if not memory_path.is_file():
        raise FileNotFoundError(f"Missing memory file: {memory_path}")
    memory = load_json(memory_path)
    if not isinstance(memory, dict) or not isinstance(memory.get("events"), list):
        raise ValueError(f"Invalid memory file: {memory_path}")

    video = memory.get("video")
    fps = float(video.get("fps", 0)) if isinstance(video, dict) else 0.0
    if not math.isfinite(fps) or fps < 0:
        raise ValueError(f"Invalid FPS in {memory_path}: {fps!r}")

    events: list[dict[str, Any]] = []
    for source_position, event in enumerate(memory["events"]):
        if not isinstance(event, dict):
            raise TypeError(f"Event {source_position} in {memory_path} is not an object")
        start_sec, end_sec = event_interval_seconds(event)
        key_frame = event.get("key_frame")
        if fps > 0 and isinstance(key_frame, int) and key_frame >= 0:
            key_timestamp_sec = key_frame / fps
        else:
            key_timestamp_sec = (start_sec + end_sec) / 2.0
        events.append(
            {
                "id": event.get("id", source_position),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "key_timestamp_sec": key_timestamp_sec,
                "keyframe": resolve_keyframe(event.get("frame_dir"), path_root, memory_path),
            }
        )

    events.sort(key=lambda item: (item["start_sec"], item["end_sec"], str(item["id"])))
    if not events:
        raise ValueError(f"No events in {memory_path}")
    if max_events and len(events) > max_events:
        raise ValueError(
            f"{video_uid} has {len(events)} events, exceeding --max-events {max_events}; "
            "no event was dropped"
        )
    return memory_path, events


def build_timeline(events: list[dict[str, Any]]) -> str:
    lines = []
    for frame_number, event in enumerate(events, start=1):
        lines.append(
            f"Frame {frame_number}: event {event['id']}; "
            f"event interval [{format_timestamp(event['start_sec'])}, "
            f"{format_timestamp(event['end_sec'])}]; "
            f"keyframe timestamp {format_timestamp(event['key_timestamp_sec'])}."
        )
    return "\n".join(lines)


def build_prompt(record: dict[str, Any], timeline: str, event_count: int) -> str:
    choices = [str(choice).strip() for choice in record["choices"]]
    labels = ", ".join(LETTERS[: len(choices)])
    options = "\n".join(
        f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices)
    )
    return (
        "<video>\n"
        f"The video input contains {event_count} chronologically ordered event keyframes. "
        "Each frame is one event's representative keyframe. Use this exact mapping:\n"
        f"{timeline}\n"
        "Select the best answer using only these keyframes and timestamps. "
        f"Respond with exactly one letter ({labels}) and no explanation.\n"
        f"Question: {record['question']}\n"
        f"{options}\n"
        "The best answer is:"
    )


def extract_answer(text: str, option_count: int) -> str | None:
    allowed = LETTERS[:option_count]
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


def media_uuid(memory_path: Path, events: list[dict[str, Any]], image_max_pixels: int) -> str:
    memory_stat = memory_path.stat()
    parts = [str(memory_path.resolve()), str(memory_stat.st_size), str(memory_stat.st_mtime_ns)]
    parts.append(str(image_max_pixels))
    for event in events:
        path = event["keyframe"]
        stat = path.stat()
        parts.extend((str(path), str(stat.st_size), str(stat.st_mtime_ns)))
    return "event-keyframes-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()

    # Keep metadata/path inspection usable on CPU-only machines. Heavy runtime
    # dependencies are needed only after command-line parsing reaches inference.
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.extras.packages import is_vllm_available
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    if not is_vllm_available():
        raise RuntimeError("vLLM is not installed in the current environment")
    from vllm import LLM, SamplingParams

    if args.request_batch_size < 1 or args.cutoff_len < 1 or args.max_new_tokens < 1:
        raise ValueError("Batch size and token limits must be positive")
    if args.max_events < 0:
        raise ValueError("--max-events cannot be negative")
    if not args.memory_dir.is_dir():
        raise NotADirectoryError(args.memory_dir)

    questions = load_questions(args.questions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in questions:
        grouped[str(record["video_uid"])].append(record)

    model_args, data_args, _, generating_args = get_infer_args(
        {
            "model_name_or_path": args.model_name_or_path,
            "template": args.template,
            "cutoff_len": args.cutoff_len,
            "enable_thinking": False,
            "vllm_config": args.vllm_config,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    template.mm_plugin.expand_mm_tokens = False

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": args.cutoff_len + args.max_new_tokens,
        "tensor_parallel_size": 1,
        "disable_log_stats": True,
        "limit_mm_per_prompt": {"image": 0, "video": 1, "audio": 0},
    }
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k or -1,
        max_tokens=generating_args.max_new_tokens,
        stop_token_ids=template.get_stop_token_ids(tokenizer),
        seed=args.seed,
    )

    args.save_name.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.save_name.with_suffix(args.save_name.suffix + ".partial")
    progress_path = args.save_name.with_suffix(args.save_name.suffix + ".progress.json")
    summary_path = args.save_name.with_suffix(args.save_name.suffix + ".summary.json")
    results: list[dict[str, Any] | None] = [None] * len(questions)
    event_counts: dict[str, int] = {}
    completed = parsed = correct = 0
    started = time.perf_counter()

    with partial_path.open("w", encoding="utf-8", newline="\n") as partial_file:
        for video_number, video_uid in enumerate(tqdm(sorted(grouped), desc="Videos"), start=1):
            memory_path, events = load_event_bank(
                video_uid,
                args.memory_dir,
                args.path_root,
                args.max_events,
            )
            event_counts[video_uid] = len(events)
            frame_paths = [event["keyframe"].as_posix() for event in events]
            regularized = template.mm_plugin._regularize_videos(
                [frame_paths],
                image_max_pixels=args.image_max_pixels,
                image_min_pixels=args.image_min_pixels,
                video_fps=1.0,
                video_maxlen=len(frame_paths),
            )["videos"]
            metadata_kwargs = {"do_sample_frames": False, "total_num_frames": len(frame_paths)}
            video_metadata = {
                "fps": 1.0,
                "frames_indices": list(range(len(frame_paths))),
                "total_num_frames": len(frame_paths),
                "video_backend": "opencv",
            }
            multimodal_data = {"video": (regularized, video_metadata)}
            video_uuid = media_uuid(memory_path, events, args.image_max_pixels)
            timeline = build_timeline(events)
            video_questions = grouped[video_uid]

            for start in range(0, len(video_questions), args.request_batch_size):
                batch = video_questions[start : start + args.request_batch_size]
                inputs = []
                for record in batch:
                    prompt = build_prompt(record, timeline, len(events))
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ""},
                    ]
                    messages = template.mm_plugin.process_messages(
                        messages,
                        [],
                        [frame_paths],
                        [],
                        processor,
                    )
                    prompt_ids, _ = template.encode_oneturn(tokenizer, messages)
                    prompt_ids, _ = template.mm_plugin.process_token_ids(
                        prompt_ids,
                        None,
                        [],
                        [frame_paths],
                        [],
                        tokenizer,
                        processor,
                    )
                    if len(prompt_ids) + args.max_new_tokens > args.cutoff_len:
                        raise ValueError(
                            f"Prompt for video_uid={video_uid}, qid={record['qid']} has "
                            f"{len(prompt_ids)} text tokens, exceeding cutoff_len={args.cutoff_len}"
                        )
                    inputs.append(
                        {
                            "prompt_token_ids": prompt_ids,
                            "multi_modal_data": multimodal_data,
                            "multi_modal_uuids": {"video": [video_uuid]},
                            "mm_processor_kwargs": metadata_kwargs,
                        }
                    )

                outputs = llm.generate(inputs, sampling_params)
                for record, output in zip(batch, outputs):
                    prediction = output.outputs[0].text
                    parsed_answer = extract_answer(prediction, len(record["choices"]))
                    is_correct = parsed_answer == record["answer"]
                    result = {
                        **record,
                        "event_count": len(events),
                        "predict": prediction,
                        "parsed_answer": parsed_answer,
                        "correct": is_correct,
                    }
                    results[int(record["row_index"])] = result
                    partial_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed += 1
                    parsed += int(parsed_answer is not None)
                    correct += int(is_correct)
                partial_file.flush()

            elapsed = time.perf_counter() - started
            progress = {
                "completed_questions": completed,
                "total_questions": len(questions),
                "completed_videos": video_number,
                "total_videos": len(grouped),
                "current_video_uid": video_uid,
                "current_event_count": len(events),
                "samples_per_second": completed / elapsed if elapsed > 0 else 0.0,
                "parsed": parsed,
                "correct": correct,
                "accuracy": correct / completed if completed else None,
            }
            write_json_atomic(progress_path, progress)
            del regularized, multimodal_data
            gc.collect()

    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions; first missing row_index={missing[0]}")
    with args.save_name.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    runtime = time.perf_counter() - started
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "questions": len(questions),
        "videos": len(grouped),
        "runtime_seconds": runtime,
        "samples_per_second": len(questions) / runtime if runtime > 0 else 0.0,
        "total_unique_event_keyframes": sum(event_counts.values()),
        "min_events_per_video": min(event_counts.values()),
        "max_events_per_video": max(event_counts.values()),
        "mean_events_per_video": sum(event_counts.values()) / len(event_counts),
        "parsed": parsed,
        "parse_failures": len(questions) - parsed,
        "correct": correct,
        "accuracy": correct / len(questions),
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "perception_text_included": False,
        "timestamps_in_prompt": True,
        "predictions": str(args.save_name),
        "partial_predictions": str(partial_path),
        "progress": str(progress_path),
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
