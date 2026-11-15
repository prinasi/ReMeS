#!/usr/bin/env python3
"""Geometry training / evaluation runner for ReMeS (T²-Splat stage).

Dataset presets live in configs/<dataset>.yaml on top of configs/common.yaml.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("geometry", "texture", "render", "metrics", "export")
DATASETS = tuple(sorted(p.stem for p in (ROOT / "configs").glob("*.yaml") if p.stem != "common"))


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader with scientific notation for learning rates."""


# PyYAML otherwise treats values such as 1e-3 as strings.
ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def read_config(path):
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=ConfigLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration {path}: {error}") from error
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML configuration must contain a mapping: {path}")
    for section, values in config.items():
        if not isinstance(section, str) or not isinstance(values, dict):
            raise ValueError(f"Configuration section {section!r} must be a named mapping: {path}")
    return config


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def cli_options(settings):
    # Legacy arguments infer their type from defaults, including integer zero.
    def cli_value(value):
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)

    arguments = []
    for key, value in settings.items():
        if value is None or value is False:
            continue
        arguments.append("--" + key)
        if value is not True:
            arguments.extend(cli_value(v) for v in value) if isinstance(value, list) else arguments.append(cli_value(value))
    return arguments


def apply_setting(config, setting):
    key, separator, raw = setting.partition("=")
    if not separator or "." not in key:
        raise ValueError("Use --set section.option=value")
    section, name = key.split(".", 1)
    if section not in {"training", "texture", "pipeline", "data"} or not name or "." in name:
        raise ValueError(f"Unknown setting: {key}")
    try:
        value = yaml.load(raw, Loader=ConfigLoader)
    except yaml.YAMLError:
        value = raw
    if section != "training" and name not in config.get(section, {}):
        raise ValueError(f"Unknown setting: {key}")
    config.setdefault(section, {})[name] = value


def load_config(args):
    config = read_config(ROOT / "configs/common.yaml")
    config = merge(config, read_config(ROOT / f"configs/{args.dataset}.yaml"))
    if args.config:
        config = merge(config, read_config(absolute(args.config)))
    # CLI settings override environment variables.
    env_settings = {"DATA_ROOT": "data.root", "OUTPUT_ROOT": "data.output",
                    "FACE_TEX_RES": "texture.face_tex_res", "RDT_BACKEND": "training.rdt_backend",
                    "VISIBILITY_PRUNING": "training.visibility_pruning",
                    "HEARTBEAT_SAFETY": "training.heartbeat_safety",
                    "TOPOLOGY_MONITOR_INTERVAL": "training.topology_monitor_interval"}
    for variable, key in env_settings.items():
        if variable in os.environ:
            apply_setting(config, key + "=" + os.environ[variable])
    for setting in args.set:
        apply_setting(config, setting)
    for arg, key in ((args.data_root, "root"), (args.output_root, "output")):
        if arg is not None:
            config["data"][key] = arg
    if args.face_tex_res is not None:
        config["texture"]["face_tex_res"] = args.face_tex_res
    if args.skip_texture:
        config["pipeline"]["texture"] = False
    if args.no_mip:
        config["texture"]["use_mip_texture"] = False
    if args.texture_offload:
        config["texture"]["texture_adam_offload"] = True
    if int(config["training"]["iterations"]) <= 0 or int(config["texture"]["stage2_iterations"]) <= 0:
        raise ValueError("Iteration counts must be positive")
    if int(config["texture"]["face_tex_res"]) < 1:
        raise ValueError("Texture resolution must be positive")
    return config


def selected_scenes(config, value):
    names = list(config["scenes"]) if value == "all" else value.split(",")
    if not names or any(name not in config["scenes"] for name in names):
        raise ValueError(f"Unknown scene selection {value!r}; choices: {', '.join(config['scenes'])}")
    return list(dict.fromkeys(names))


def make_plan(args, config, name):
    scene = merge(config.get("scene_defaults", {}), config["scenes"][name])
    kind = scene["kind"]
    training = merge(config["training"], config.get(kind, {}))
    if args.reg_mode == "strong":
        training = merge(training, config.get("strong", {}).get(kind, {}))
    training = merge(training, scene.get("training", {}))
    training["max_points"] = scene["max_points"]
    # An explicit training override must win over indoor/outdoor defaults.
    for setting in args.set:
        if setting.startswith("training."):
            holder = {"training": training}
            apply_setting(holder, setting)
            training = holder["training"]
    final_threshold = training.get("final_visibility_threshold", 0.5)
    if (type(final_threshold) not in (int, float) or not math.isfinite(final_threshold)
            or not 0 <= final_threshold <= 1):
        raise ValueError("training.final_visibility_threshold must be finite and within [0, 1]")
    source = absolute(config["data"]["root"]) / config["data"]["layout"].format(scene=name)
    output = absolute(config["data"]["output"]) / name
    geometry_iter = int(training["iterations"])
    pipeline = merge(config["pipeline"], scene.get("pipeline", {}))
    # Explicit CLI pipeline settings also override scene-specific postprocessing.
    for setting in args.set:
        if setting.startswith("pipeline."):
            holder = {"pipeline": pipeline}
            apply_setting(holder, setting)
            pipeline = holder["pipeline"]
    if args.skip_texture:
        pipeline["texture"] = False
    mesh_ratio = pipeline.get("mesh_min_faces_ratio", 0.01)
    if (type(mesh_ratio) not in (int, float) or not math.isfinite(mesh_ratio)
            or not 0 <= mesh_ratio <= 1):
        raise ValueError("pipeline.mesh_min_faces_ratio must be finite and within [0, 1]")
    texture = config["texture"]
    final_iter = geometry_iter + (int(texture["stage2_iterations"]) if pipeline["texture"] else 0)
    init_model = absolute(args.init_from) / name if args.init_from else output
    checkpoint = lambda parent, iteration: parent / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    plan = []

    def add(stage, script, arguments, requires=(), produces=(), cwd=ROOT):
        plan.append({"stage": stage, "argv": [sys.executable, "-u", str(ROOT / script), *map(str, arguments)],
                     "cwd": str(cwd), "requires": list(map(str, requires)), "produces": list(map(str, produces))})

    save_iterations = training.get("save_iterations", [])
    if not isinstance(save_iterations, list) or any(
            type(value) is not int or not 1 <= value <= geometry_iter for value in save_iterations):
        raise ValueError("training.save_iterations must be a list of integers within the training interval")
    training.update(eval=True, save_iterations=sorted(set(save_iterations + [geometry_iter])),
                    test_iterations=[geometry_iter])
    image_args = ["-i", scene["images"]] if scene.get("images") else []
    add("geometry", "train.py", ["-s", source, *image_args, "-m", output, *cli_options(training)],
        requires=[source], produces=[checkpoint(output, geometry_iter)])
    if pipeline["texture"]:
        arguments = ["-m", output, "--iteration", geometry_iter, *cli_options(texture)]
        if args.init_from:
            arguments += ["--init_from", init_model]
        add("texture", "train_face_texture.py", arguments,
            requires=[checkpoint(init_model, geometry_iter), init_model / "cfg_args"], produces=[checkpoint(output, final_iter)])
    if pipeline["image_metrics"]:
        add("render", "render.py", ["-m", output, "--iteration", final_iter, "--skip_train", "--eval", "--quiet"],
            requires=[checkpoint(output, final_iter)], produces=[output / "test" / f"ours_{final_iter}" / "renders"])
        add("metrics", "metrics.py", ["-m", output], requires=[output / "test" / f"ours_{final_iter}" / "renders"],
            produces=[output / "results.json"])
    mesh = output / "mesh.ply"
    if pipeline["export_mesh"]:
        # Bake full SH and MipTex into the original vertex RGB without adding triangles.
        export_options = {"clean": pipeline["mesh_clean"], "manifold": pipeline["mesh_manifold"],
                          "overwrite": True}
        if pipeline["mesh_clean"]:
            export_options["min_faces_ratio"] = mesh_ratio
        add("export", "create_ply.py", [checkpoint(output, final_iter).parent, "--out", mesh,
                                        *cli_options(export_options)],
            requires=[checkpoint(output, final_iter)], produces=[mesh])
    if args.only_stage:
        plan = [step for step in plan if step["stage"] == args.only_stage]
    else:
        plan = [step for step in plan if STAGES.index(step["stage"]) >= STAGES.index(args.start_at)]
    if not plan:
        raise ValueError("No enabled stages selected; inspect the pipeline settings")
    if args.init_from and (not pipeline["texture"] or plan[0]["stage"] != "texture"):
        raise ValueError("--init-from requires starting at the enabled texture stage")
    return output, final_iter, plan


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def mesh_counts(path):
    """Read final exported topology counts without loading the full mesh."""
    counts = {}
    with Path(path).open("rb") as stream:
        for _ in range(200):
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                counts["mesh_vertices"] = int(line.split()[-1])
            elif line.startswith("element face "):
                counts["mesh_faces"] = int(line.split()[-1])
            elif line == "end_header":
                break
        else:
            raise ValueError(f"Invalid PLY header: {path}")
    if set(counts) != {"mesh_vertices", "mesh_faces"} or min(counts.values()) <= 0:
        raise ValueError(f"Empty or invalid exported mesh: {path}")
    return counts


def run_stage(step, output, env):
    for required in step["requires"]:
        if not Path(required).exists():
            raise FileNotFoundError(f"{step['stage']}: required input not found: {required}")
    log_path = output / f"{step['stage']}.log"
    print(f"[{step['stage']}] {shlex.join(step['argv'])}\nLog: {log_path}", flush=True)
    with log_path.open("a") as log:
        proc = subprocess.Popen(step["argv"], cwd=step["cwd"], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, step["argv"])
    for produced in step["produces"]:
        path = Path(produced)
        if not path.exists() or (path.is_file() and not path.stat().st_size) or (path.is_dir() and not any(path.iterdir())):
            raise RuntimeError(f"{step['stage']}: missing or empty output: {path}")


def execute_scene(args, config, name):
    output, final_iter, plan = make_plan(args, config, name)
    if args.dry_run:
        print(json.dumps({"scene": name, "output": str(output), "final_iteration": final_iter, "steps": plan}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental replacement of a completed scene; continuation is explicit.
    if plan[0]["stage"] == "geometry" and (output / "point_cloud").exists():
        raise FileExistsError(f"Existing model in {output}; use a new output root or --start-at")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", SETUPTOOLS_USE_DISTUTILS="stdlib")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "submodules/effrdel/src"),
                                       str(ROOT / "submodules/diff-triangle-mesh-rasterization"),
                                       str(ROOT / "submodules/simple-knn"), str(ROOT / "submodules/nvdiffrast"), env.get("PYTHONPATH", "")])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    record = {"status": "running", "scene": name, "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "final_iteration": final_iter, "steps": plan, "completed_stages": []}
    status_path = output / "pipeline_status.json"
    if status_path.exists():
        previous = output / ("pipeline_status_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".json")
        previous.write_bytes(status_path.read_bytes())
    write_json(status_path, record)
    try:
        for step in plan:
            record["active_stage"] = step["stage"]
            write_json(status_path, record)
            run_stage(step, output, env)
            if step["stage"] == "export":
                counts = mesh_counts(output / "mesh.ply")
                record.update(counts)
                write_json(output / "triangle_counts.json", counts)
            if step["stage"] == "metrics":
                result = json.loads((output / "results.json").read_text())[f"ours_{final_iter}"]
                if not all(math.isfinite(result[key]) for key in ("PSNR", "SSIM", "LPIPS")):
                    raise RuntimeError("Non-finite image quality metrics")
                record["metrics"] = result
            record["completed_stages"].append(step["stage"])
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, record)
    print(f"Completed {name}: {output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=DATASETS, default="synthetic4relight")
    result.add_argument("--gpu", default="0")
    result.add_argument("--scenes", default="all", help="Comma-separated scene names, or all")
    result.add_argument("--reg-mode", choices=("default", "strong"), default="default")
    result.add_argument("--config", help="Additional YAML overrides (.yaml or .yml)")
    result.add_argument("--data-root")
    result.add_argument("--output-root")
    result.add_argument("--face-tex-res", type=int)
    result.add_argument("--skip-texture", action="store_true")
    result.add_argument("--no-mip", action="store_true")
    result.add_argument("--texture-offload", action="store_true")
    result.add_argument("--start-at", choices=STAGES, default="geometry")
    result.add_argument("--only-stage", choices=STAGES)
    result.add_argument("--init-from", help="Existing output ROOT; load each scene's geometry without retraining it")
    result.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    result.add_argument("--dry-run", action="store_true", help="Print commands without CUDA, data checks, or file writes")
    return result


def main():
    args = parser().parse_args()
    try:
        config = load_config(args)
        for name in selected_scenes(config, args.scenes):
            execute_scene(args, config, name)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, KeyError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
