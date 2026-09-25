"""
MAURICE Unified Command Line Interface (maurice/cli.py)
"""

import argparse
import platform
import subprocess
import sys

import maurice
from maurice.eval import main as eval_main
from maurice.merge import main as merge_main
from maurice.prepare import main as prepare_main
from maurice.serve import main as serve_main
from maurice.train import main as train_main


def run_quantize(variant: str, quant_type: str = "q4_k_m"):
    """Invokes scripts/04_quantize_imatrix.sh via subprocess."""
    cmd = ["bash", "scripts/04_quantize_imatrix.sh", variant, quant_type]
    print(f"Running quantization: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    sys.exit(res.returncode)


def run_ui():
    """Launches Streamlit UI."""
    cmd = ["streamlit", "run", "ui/app.py"]
    print(f"Launching Streamlit UI: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    sys.exit(res.returncode)


def print_info():
    """Prints installed package version, hardware detection, and available variants."""
    try:
        from maurice.eval import detect_hardware_accel

        hw = detect_hardware_accel()
    except Exception:  # noqa: BLE001
        hw = f"{platform.system()} {platform.machine()}"

    print(f"MAURICE Package Version: {maurice.__version__}")
    print(f"Author: {maurice.__author__}")
    print(f"Detected Hardware Acceleration: {hw}")
    print("Available Model Variants:")
    print("  - c: mau-llm-1.0-c (Code & Refactor)")
    print("  - r: mau-llm-1.0-r (Pure Reasoning)")
    print("  - g: mau-llm-1.0-g (General Purpose)")


def main(args_list: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="maurice",
        description="MAURICE: Minimal Adaptation for Ultra-fast Reasoning and Inference in Code Engines",
    )
    parser.add_argument("--version", action="version", version=f"maurice {maurice.__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # prepare
    prep_parser = subparsers.add_parser("prepare", help="Prepare datasets for target variants")
    prep_parser.add_argument("--variant", choices=["c", "r", "g", "all"], default="all")
    prep_parser.add_argument("--sample-size", type=int, default=50)
    prep_parser.add_argument("--synthetic", action="store_true", default=True)
    prep_parser.add_argument("--real-hf", action="store_false", dest="synthetic")
    prep_parser.add_argument("--dry-run", action="store_true")

    # train
    train_parser = subparsers.add_parser("train", help="Train QLoRA adapters")
    train_parser.add_argument("--variant", choices=["c", "r", "g"], required=True)
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--output-dir", type=str, default=None)

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapters into base FP16 model")
    merge_parser.add_argument("--variant", choices=["c", "r", "g"], required=True)
    merge_parser.add_argument("--adapter-path", type=str, default=None)
    merge_parser.add_argument("--output-dir", type=str, default=None)
    merge_parser.add_argument("--save-method", type=str, default="merged_16bit")
    merge_parser.add_argument("--dry-run", action="store_true")

    # quantize
    quant_parser = subparsers.add_parser("quantize", help="Convert to GGUF and quantize with imatrix")
    quant_parser.add_argument("--variant", choices=["c", "r", "g", "all"], default="c")
    quant_parser.add_argument("--quant-type", type=str, default="q4_k_m")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run hardware benchmark and evaluation harness")
    eval_parser.add_argument("--variant", choices=["c", "r", "g", "all"], default="all")
    eval_parser.add_argument("--model-path", type=str, default=None)
    eval_parser.add_argument("--output-json", type=str, default="build/benchmark_results.json")
    eval_parser.add_argument("--dry-run", action="store_true")
    eval_parser.add_argument("--input-json", type=str, default=None)
    eval_parser.add_argument("--analyze", action="store_true")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI inference server")
    serve_parser.add_argument("--variant", choices=["c", "r", "g"], default="c")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--backend", choices=["hf", "mock"], default="hf")

    # ui
    subparsers.add_parser("ui", help="Launch Streamlit UI")

    # info
    subparsers.add_parser("info", help="Display version, hardware, and environment info")

    parsed_args, _extra = parser.parse_known_args(args_list)

    if not parsed_args.subcommand:
        parser.print_help()
        sys.exit(0)

    sub = parsed_args.subcommand

    # Forward to specific modules or actions
    if sub == "prepare":
        prep_args = []
        if parsed_args.variant:
            prep_args.extend(["--variant", parsed_args.variant])
        if parsed_args.sample_size:
            prep_args.extend(["--sample-size", str(parsed_args.sample_size)])
        if parsed_args.dry_run:
            prep_args.append("--dry-run")
        prep_args.append("--synthetic" if parsed_args.synthetic else "--real-hf")
        prepare_main(prep_args)

    elif sub == "train":
        t_args = ["--variant", parsed_args.variant]
        if parsed_args.config:
            t_args.extend(["--config", parsed_args.config])
        if parsed_args.dry_run:
            t_args.append("--dry-run")
        if parsed_args.output_dir:
            t_args.extend(["--output-dir", parsed_args.output_dir])
        train_main(t_args)

    elif sub == "merge":
        m_args = ["--variant", parsed_args.variant]
        if parsed_args.adapter_path:
            m_args.extend(["--adapter-path", parsed_args.adapter_path])
        if parsed_args.output_dir:
            m_args.extend(["--output-dir", parsed_args.output_dir])
        if parsed_args.save_method:
            m_args.extend(["--save-method", parsed_args.save_method])
        if parsed_args.dry_run:
            m_args.append("--dry-run")
        merge_main(m_args)

    elif sub == "quantize":
        run_quantize(parsed_args.variant, parsed_args.quant_type)

    elif sub == "eval":
        e_args = []
        if parsed_args.variant:
            e_args.extend(["--variant", parsed_args.variant])
        if parsed_args.model_path:
            e_args.extend(["--model-path", parsed_args.model_path])
        if parsed_args.output_json:
            e_args.extend(["--output-json", parsed_args.output_json])
        if parsed_args.dry_run:
            e_args.append("--dry-run")
        if parsed_args.input_json:
            e_args.extend(["--input-json", parsed_args.input_json])
        if parsed_args.analyze:
            e_args.append("--analyze")
        eval_main(e_args)

    elif sub == "serve":
        s_args = [
            "--variant",
            parsed_args.variant,
            "--host",
            parsed_args.host,
            "--port",
            str(parsed_args.port),
            "--backend",
            parsed_args.backend,
        ]
        serve_main(s_args)

    elif sub == "ui":
        run_ui()

    elif sub == "info":
        print_info()


if __name__ == "__main__":
    main()
