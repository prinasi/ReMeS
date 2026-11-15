"""Runner contract tests (Synthetic4Relight preset + generic config handling)."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def args_config(self, *extra):
        args = pipeline.parser().parse_args(list(extra))
        with patch.dict(os.environ, {}, clear=True):
            config = pipeline.load_config(args)
        return args, config

    def test_datasets_are_discovered_from_configs(self):
        self.assertIn("synthetic4relight", pipeline.DATASETS)
        self.assertNotIn("common", pipeline.DATASETS)

    def test_shell_entry_outside_project_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="t2 pipeline ") as directory:
            output = Path(directory) / "new output"
            env = os.environ.copy()
            env["PYTHON"] = sys.executable
            result = subprocess.run(["bash", str(pipeline.ROOT / "scripts/run.sh"), "synthetic4relight",
                                     "1", "hotdog", "default", "nomip", "--dry-run",
                                     "--output-root", str(output)], cwd=directory, env=env,
                                    text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["output"], str(output / "hotdog"))
            self.assertEqual(plan["final_iteration"], 35000)
            texture = next(s for s in plan["steps"] if s["stage"] == "texture")
            self.assertNotIn("--use_mip_texture", texture["argv"])
            self.assertFalse(output.exists())

    def test_synthetic4relight_preset(self):
        import argparse
        from arguments import ModelParams, OptimizationParams, PipelineParams
        parser = argparse.ArgumentParser()
        ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
        for name in ("test_iterations", "save_iterations"):
            parser.add_argument("--" + name, nargs="+", type=int)
        args, config = self.args_config()
        self.assertEqual(set(pipeline.selected_scenes(config, "all")), {"air_baloons", "chair", "hotdog", "jugs"})
        _, final, steps = pipeline.make_plan(args, config, "jugs")
        self.assertEqual([s["stage"] for s in steps], ["geometry", "texture", "render", "metrics", "export"])
        self.assertEqual(final, 35000)
        argv = steps[0]["argv"]
        self.assertNotIn("-i", argv)                       # Blender-style layout: no image folder
        opts = parser.parse_args(argv[3:])
        self.assertEqual(opts.source_path, str(pipeline.absolute("data/Synthetic4Relight") / "jugs"))
        self.assertTrue(opts.random_background and opts.disable_priors and opts.eval)
        self.assertEqual(opts.lambda_mask, 0.1)
        # the exported mesh must be opaque and hard-edged by the end of training
        self.assertLessEqual(opts.final_opacity_iter, opts.iterations)
        self.assertLessEqual(opts.sigma_until, opts.iterations)
        self.assertEqual(opts.save_iterations, [30000])

    def test_yaml_overrides_preserve_types_and_merge_defaults(self):
        for suffix in (".yaml", ".yml"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("overrides" + suffix)
                path.write_text(
                    "data:\n  root: '/tmp/custom data'\n"
                    "texture:\n  face_tex_res: 16\n  use_mip_texture: false\n"
                    "  lr_face_texture: 1e-3\n"
                    "scenes:\n  custom:\n    kind: outdoor\n    images: images\n"
                    "    max_points: 123456\n")
                args, config = self.args_config("--config", str(path), "--scenes", "custom")
                self.assertEqual(config["texture"]["lr_face_texture"], .001)
                self.assertIsInstance(config["texture"]["lr_face_texture"], float)
                self.assertIs(config["texture"]["use_mip_texture"], False)
                self.assertEqual(config["training"]["iterations"], 30000)
                _, final, steps = pipeline.make_plan(args, config, "custom")
                self.assertEqual(final, 35000)
                self.assertIn("/tmp/custom data/custom", steps[0]["argv"])
                self.assertIn("-i", steps[0]["argv"])
                self.assertNotIn("--use_mip_texture", steps[1]["argv"])

    def test_yaml_cli_values_and_quoted_strings(self):
        _, config = self.args_config(
            "--set", "texture.lr_face_texture=1e-3",
            "--set", "texture.use_mip_texture=false",
            "--set", "training.test_iterations=[1000, 2000]",
            "--set", 'data.root="off"',
            "--set", "training.example=null")
        self.assertEqual(config["texture"]["lr_face_texture"], .001)
        self.assertIsInstance(config["texture"]["lr_face_texture"], float)
        self.assertIs(config["texture"]["use_mip_texture"], False)
        self.assertEqual(config["training"]["test_iterations"], [1000, 2000])
        self.assertEqual(config["data"]["root"], "off")
        self.assertIsNone(config["training"]["example"])

    def test_invalid_yaml_and_python_tags_are_rejected(self):
        cases = [
            ("training: [", "Invalid YAML"),
            ("[training, texture]", "must contain a mapping"),
            ("training: false", "must be a named mapping"),
            ("training: !!python/object/apply:builtins.dict []", "Invalid YAML"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            for content, message in cases:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(ValueError, message):
                        self.args_config("--config", str(path))

    def test_empty_yaml_override_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.yaml"
            path.write_text("# No overrides\n")
            _, config = self.args_config("--config", str(path))
            self.assertEqual(config["texture"]["face_tex_res"], 8)
            self.assertEqual(config["training"]["iterations"], 30000)

    def test_float_zero_is_accepted_by_legacy_integer_defaults(self):
        import argparse
        from arguments import OptimizationParams
        parser = argparse.ArgumentParser()
        OptimizationParams(parser)
        args = parser.parse_args(pipeline.cli_options({"lambda_vertex_normal": 0.0}))
        self.assertEqual(args.lambda_vertex_normal, 0)

    def test_explicit_training_override_wins(self):
        args, config = self.args_config("--scenes", "hotdog", "--set", "training.heartbeat_collapse_ratio=0.03")
        _, _, steps = pipeline.make_plan(args, config, "hotdog")
        argv = steps[0]["argv"]
        self.assertEqual(argv[argv.index("--heartbeat_collapse_ratio") + 1], "0.03")

    def test_intermediate_mesh_save_schedule_reaches_train_command(self):
        args, config = self.args_config(
            "--set", "training.save_iterations=[20000,12000,20000]",
            "--set", "training.save_topology_snapshots=true")
        _, _, steps = pipeline.make_plan(args, config, "hotdog")
        argv = steps[0]["argv"]
        start = argv.index("--save_iterations") + 1
        end = next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))
        self.assertEqual(argv[start:end], ["12000", "20000", "30000"])
        self.assertIn("--save_topology_snapshots", argv)
        self.assertIn("iteration_30000", steps[0]["produces"][0])

    def test_invalid_intermediate_mesh_save_schedule_rejected(self):
        for value in ("[0]", "[30001]", "[1.5]", "[true]", "20000", "null"):
            with self.subTest(value=value):
                args, config = self.args_config("--set", "training.save_iterations=" + value)
                with self.assertRaisesRegex(ValueError, "save_iterations"):
                    pipeline.make_plan(args, config, "hotdog")

    def test_mesh_clean_override(self):
        for clean in (True, False):
            args, config = self.args_config("--set", "pipeline.mesh_clean=" + str(clean).lower(),
                                            "--set", "pipeline.mesh_min_faces_ratio=0.005")
            _, _, steps = pipeline.make_plan(args, config, "hotdog")
            export = next(s["argv"] for s in steps if s["stage"] == "export")
            self.assertEqual("--clean" in export, clean)
            if clean:
                self.assertEqual(float(export[export.index("--min_faces_ratio") + 1]), .005)

    def test_invalid_mesh_component_ratio_rejected_before_training(self):
        for value in ("-0.1", "1.1", ".nan", ".inf", "true", "null", '"bad"'):
            with self.subTest(value=value):
                args, config = self.args_config("--set", "pipeline.mesh_min_faces_ratio=" + value)
                with self.assertRaisesRegex(ValueError, "mesh_min_faces_ratio"):
                    pipeline.make_plan(args, config, "hotdog")

    def test_invalid_final_visibility_threshold_rejected_before_training(self):
        for value in ("-0.1", "1.1", ".nan", ".inf", "true", "null", '"bad"'):
            with self.subTest(value=value):
                args, config = self.args_config("--set", "training.final_visibility_threshold=" + value)
                with self.assertRaisesRegex(ValueError, "final_visibility_threshold"):
                    pipeline.make_plan(args, config, "hotdog")

    def test_independent_texture_run_and_offload(self):
        args, config = self.args_config("--start-at", "texture", "--init-from", "output/base",
                                       "--output-root", "output/new", "--face-tex-res", "16", "--texture-offload")
        output, final, steps = pipeline.make_plan(args, config, "hotdog")
        self.assertEqual(output, pipeline.ROOT / "output/new/hotdog")
        self.assertEqual(final, 35000)
        self.assertEqual(steps[0]["stage"], "texture")
        self.assertIn(str(pipeline.ROOT / "output/base/hotdog"), steps[0]["argv"])
        self.assertIn("--texture_adam_offload", steps[0]["argv"])
        self.assertNotIn("geometry", [s["stage"] for s in steps])

    def test_init_from_cannot_retrain_geometry(self):
        args, config = self.args_config("--init-from", "output/base")
        with self.assertRaises(ValueError):
            pipeline.make_plan(args, config, "hotdog")

    def test_unknown_scene_fails_before_execution(self):
        args, config = self.args_config()
        with self.assertRaises(ValueError):
            pipeline.selected_scenes(config, "hotdog,typo")

    def run_fixture(self, directory, code, produces=(), extra_steps=()):
        args, config = self.args_config()
        output = Path(directory) / "run"
        plan = [{"stage": "geometry", "argv": [sys.executable, "-c", code], "cwd": directory,
                 "requires": [], "produces": list(map(str, produces))}, *extra_steps]
        with patch.object(pipeline, "make_plan", return_value=(output, 30000, plan)), contextlib.redirect_stdout(io.StringIO()):
            pipeline.execute_scene(args, config, "hotdog")
        return output

    def test_subprocess_failure_stops_later_stages_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must_not_exist"
            later = {"stage": "texture", "argv": [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('bad')"],
                     "cwd": directory, "requires": [], "produces": []}
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_fixture(directory, "raise SystemExit(7)", extra_steps=[later])
            status = json.loads((Path(directory) / "run/pipeline_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["completed_stages"], [])
            self.assertFalse(marker.exists())

    def test_successful_exit_without_artifact_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "missing or empty output"):
                self.run_fixture(directory, "print('finished')", [Path(directory) / "missing.pt"])
            status = json.loads((Path(directory) / "run/pipeline_status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_success_records_completion_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.pt"
            output = self.run_fixture(directory, f"open({str(artifact)!r}, 'w').write('fixture')", [artifact])
            status = json.loads((output / "pipeline_status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["completed_stages"], ["geometry"])

    def test_existing_geometry_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run/point_cloud").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                self.run_fixture(directory, "raise SystemExit('must not execute')")


if __name__ == "__main__":
    unittest.main()
