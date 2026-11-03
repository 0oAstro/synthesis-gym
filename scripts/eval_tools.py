"""Closed-loop evaluation of an open model (optionally + LoRA) on SynthGym-v0/test-200 with tool calls.

All episodes run in lockstep: each turn, the model generates for every unfinished episode in batches,
the tool call is executed in that episode's gym, and the observation is appended as a tool message.
Writes results/episodes_<name>.csv with the same columns as run_eval.py.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from synthgym.tasks import list_tasks, task_spec
from synthgym.tool_env import SynthesisToolEnv, task_prompt

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_TURNS = 10


def import_loader():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_tools", Path(__file__).parent / "train_grpo_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tokenizer


def first_tool_call(tokenizer, text: str):
    try:
        parsed = tokenizer.parse_response(text, prefix="")
    except Exception:
        return None, text
    calls = parsed.get("tool_calls") or []
    if not calls:
        return None, parsed.get("content", text)
    call = calls[0]["function"] if "function" in calls[0] else calls[0]
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return {"name": call.get("name"), "arguments": arguments}, parsed.get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--parser-from")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--chat-template-kwargs", default="")
    parser.add_argument("--suite", default="SynthGym-v0/test-200")
    parser.add_argument("--shard", default="0/1", help="k/n: evaluate every n-th task starting at k")
    parser.add_argument("--tasks", default="", help="comma-separated task ids (overrides --suite and --shard)")
    parser.add_argument("--samples", type=int, default=1, help="episodes per task (for collecting rollouts)")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    template_kwargs = {}
    for item in filter(None, args.chat_template_kwargs.split(",")):
        key, value = item.split("=")
        template_kwargs[key] = value.lower() == "true" if value.lower() in ("true", "false") else value

    tokenizer = import_loader()(args.model, args.parser_from)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.88, max_model_len=8192,
              enable_lora=args.lora is not None, max_lora_rank=16, enable_prefix_caching=True)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=192)
    lora = LoRARequest("grpo", 1, args.lora) if args.lora else None

    shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    task_ids = list_tasks(args.suite)[shard_index::shard_count]
    if args.tasks:
        task_ids = args.tasks.split(",")
    if args.samples > 1:
        task_ids = [f"{task_id}#{k}" for task_id in task_ids for k in range(args.samples)]
    suffix = "" if shard_count == 1 else f"_shard{shard_index}of{shard_count}"
    envs = {}
    messages = {}
    tools = {}
    format_errors = {t: 0 for t in task_ids}
    for task_id in task_ids:
        env = SynthesisToolEnv()
        env.reset(task_id=task_id.split("#")[0])
        envs[task_id] = env
        messages[task_id] = task_prompt(task_id.split("#")[0])
        tools[task_id] = [env.add_precursors, env.heat, env.regrind, env.xrd, env.stop]

    for turn in range(MAX_TURNS):
        active = [t for t in task_ids if not envs[t].done]
        if not active:
            break
        texts = [tokenizer.apply_chat_template(messages[t], tools=tools[t], tokenize=False, add_generation_prompt=True, **template_kwargs) for t in active]
        outputs = llm.generate(texts, sampling, lora_request=lora, use_tqdm=False)
        for task_id, output in zip(active, outputs):
            text = output.outputs[0].text
            for stop_token in [tokenizer.eos_token, "<|im_end|>", "<|eot_id|>"]:
                if stop_token:
                    text = text.split(stop_token)[0]
            call, content = first_tool_call(tokenizer, text)
            env = envs[task_id]
            if call is None or not hasattr(env, call["name"] or "") or call["name"].startswith("_") or call["name"] in ("reset", "get_reward"):
                format_errors[task_id] += 1
                messages[task_id].append({"role": "assistant", "content": text})
                messages[task_id].append({"role": "user", "content": "Call exactly one tool."})
                if format_errors[task_id] >= 3:
                    env.get_reward()
                continue
            try:
                result = getattr(env, call["name"])(**call["arguments"])
            except TypeError as error:
                result = json.dumps({"error": f"bad arguments: {error}"})
            messages[task_id].append({"role": "assistant", "content": content or "", "tool_calls": [{"type": "function", "function": call}]})
            messages[task_id].append({"role": "tool", "name": call["name"], "content": result})
        print(f"turn {turn + 1}: {sum(not envs[t].done for t in task_ids)} episodes still running", flush=True)

    rows = []
    for task_id in task_ids:
        env = envs[task_id]
        env.get_reward()
        reward = float(env.env.total_reward)
        gym = env.env
        task = task_spec(task_id.split("#")[0])
        history = gym.history
        first_add = next((tuple(sorted(r["precursors"])) for r in history if r["op"] == "add"), None)
        heats = [r["temperature_c"] for r in history if r["op"] == "heat"]
        literature_sets = {tuple(sorted(r.precursors)) for r in task.recipes}
        matching = [r.max_temp_c for r in task.recipes if tuple(sorted(r.precursors)) == first_add]
        hottest = max(heats) if heats else None
        w_target = gym.crucible.weight_fractions().get(task.target, 0.0)
        rows.append({
            "agent": args.name, "task_id": task_id, "tier": task.tier,
            "precursor_match": first_add in literature_sets if first_add else False,
            "recipe_match": bool(first_add in literature_sets and hottest is not None and any(abs(hottest - t) <= 150 for t in matching)),
            "temp_error_c": abs(hottest - float(np.median([r.max_temp_c for r in task.recipes]))) if hottest is not None else None,
            "w_target": w_target, "success_sim": w_target >= 0.9, "total_reward": reward,
            "steps": len(history), "heats": len(heats), "xrd": sum(r["op"] == "xrd" for r in history),
            "stopped_without_xrd": not any(r["op"] == "xrd" for r in history),
            "invalid": sum(r["op"] == "invalid" for r in history), "format_errors": format_errors[task_id],
        })
    frame = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    frame.to_csv(RESULTS / f"episodes_{args.name}{suffix}.csv", index=False)
    (RESULTS / f"transcripts_{args.name}{suffix}.json").write_text(json.dumps({t: messages[t] for t in task_ids}, indent=1, default=str))
    print(frame[["precursor_match", "recipe_match", "success_sim", "w_target", "total_reward", "heats", "xrd", "format_errors"]].mean().to_string())


if __name__ == "__main__":
    main()
