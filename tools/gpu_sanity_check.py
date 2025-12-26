import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


def get_git_info():
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        dirty_check = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL
        )
        dirty = "dirty" if dirty_check != 0 else "clean"

        return {"commit": commit[:12], "status": dirty}
    except Exception:
        return {"commit": "unknown", "status": "unknown"}


def save_sanity_check_result(cuda_available: bool, device_info: dict, matmul_time: float, path: Path):
    """Save sanity check results as JSON."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": device_info,
        "matmul_4096x4096_seconds": matmul_time,
        "status": "pass" if cuda_available and matmul_time < 1.0 else "warning",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Results saved to {path.as_posix()}")


def main():
    device_info = {}
    matmul_time = 0.0

    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if not cuda_available:
            print("CUDA is not available on this system. Exiting sanity check.")
            device_info = {"name": "N/A", "vram_gb": 0, "cuda_version": "N/A"}
            # Save result even if CUDA not available
            output_path = Path("tools/gpu_sanity_check_result.json")
            save_sanity_check_result(cuda_available, device_info, matmul_time, output_path)
            return

        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        total_vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU name: {props.name}")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")

        device_info = {
            "name": props.name,
            "vram_gb": round(total_vram_gb, 2),
            "cuda_version": torch.version.cuda,
        }

        torch.cuda.synchronize()
        a = torch.randn((4096, 4096), device=device, dtype=torch.float32)
        b = torch.randn((4096, 4096), device=device, dtype=torch.float32)

        start = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        matmul_time = elapsed

        print(f"4096x4096 matmul time: {elapsed:.4f} seconds")

        # Save results
        output_path = Path("tools/gpu_sanity_check_result.json")
        save_sanity_check_result(cuda_available, device_info, matmul_time, output_path)

    except Exception as exc:
        print(f"GPU sanity check failed: {exc}")


if __name__ == "__main__":
    main()
