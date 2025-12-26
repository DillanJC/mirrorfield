import time
import torch


def main():
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if not cuda_available:
            print("CUDA is not available on this system. Exiting sanity check.")
            return

        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        total_vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU name: {props.name}")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")

        torch.cuda.synchronize()
        a = torch.randn((4096, 4096), device=device, dtype=torch.float32)
        b = torch.randn((4096, 4096), device=device, dtype=torch.float32)

        start = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"4096x4096 matmul time: {elapsed:.4f} seconds")
    except Exception as exc:
        print(f"GPU sanity check failed: {exc}")


if __name__ == "__main__":
    main()
