import json
import os
import subprocess
import unittest


class TestMauricePipeline(unittest.TestCase):
    def test_config_files(self):
        for v in ["c", "r", "g"]:
            config_path = f"configs/variant_{v}.json"
            self.assertTrue(
                os.path.exists(config_path), f"Config file {config_path} missing"
            )
            with open(config_path, "r") as f:
                data = json.load(f)
                self.assertEqual(data["variant"], v)
                self.assertIn("lora", data)
                self.assertEqual(data["lora"]["r"], 16)

    def test_prepare_datasets(self):
        res = subprocess.run(
            [
                "python3",
                "scripts/01_prepare_datasets.py",
                "--variant",
                "all",
                "--sample-size",
                "5",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        for v in ["c", "r", "g"]:
            jsonl_path = f"data/processed/train_{v}.jsonl"
            self.assertTrue(os.path.exists(jsonl_path))
            with open(jsonl_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 5)

    def test_train_qlora_dry_run(self):
        for v in ["c", "r", "g"]:
            res = subprocess.run(
                [
                    "python3",
                    "scripts/02_train_qlora.py",
                    "--variant",
                    v,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
            self.assertTrue(
                os.path.exists(f"checkpoints/adapter_{v}/adapter_config.json")
            )

    def test_merge_weights_dry_run(self):
        for v in ["c", "r", "g"]:
            res = subprocess.run(
                [
                    "python3",
                    "scripts/03_merge_weights.py",
                    "--variant",
                    v,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
            self.assertTrue(
                os.path.exists(f"checkpoints/merged_{v}/config.json")
            )

    def test_quantize_imatrix(self):
        for v in ["c", "r", "g"]:
            res = subprocess.run(
                ["bash", "scripts/04_quantize_imatrix.sh", v],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
            self.assertTrue(
                os.path.exists(f"build/mau-llm-1.0-{v}-q4_k_m.gguf")
            )

    def test_benchmark_eval(self):
        res = subprocess.run(
            ["python3", "scripts/05_benchmark_eval.py", "--variant", "all"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertTrue(os.path.exists("build/benchmark_results.json"))


if __name__ == "__main__":
    unittest.main()
