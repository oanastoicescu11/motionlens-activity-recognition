"""Worker process runner for local development."""

from app.worker.main import start_workers
from app.worker.main import worker_poll_seconds_from_env


def run_workers(num_workers: int = 2, poll_seconds: float | None = None):
    """Start worker processes for inference queue consumption.
    
    Usage:
        python run_worker.py
        python run_worker.py --workers 4
        python run_worker.py --workers 4 --store-backend redis
    """
    print(f"\n{'='*70}")
    print(f"Starting {num_workers} worker process(es)")
    print(f"{'='*70}\n")

    effective_poll_seconds = worker_poll_seconds_from_env() if poll_seconds is None else max(0.01, float(poll_seconds))
    print(f"Worker poll cadence: {effective_poll_seconds:.2f}s")

    processes = start_workers(num_workers=num_workers, poll_seconds=effective_poll_seconds)
    
    print(f"✓ {len(processes)} worker(s) started")
    print(f"  Press Ctrl+C to stop\n")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print(f"\n\nStopping {len(processes)} worker(s)...")
        for p in processes:
            p.terminate()
            p.join(timeout=2)
        print("Workers stopped.")
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run worker processes for inference")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes (default: 2)")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Worker idle poll cadence in seconds (default: 0.05 or MLIVE_WORKER_POLL_SECONDS).",
    )
    parser.add_argument(
        "--store-backend",
        choices=["memory", "redis"],
        default="memory",
        help=(
            "Store backend used by workers. Use 'redis' for shared backend/worker state "
            "across processes."
        ),
    )
    
    args = parser.parse_args()
    import os

    os.environ["MLIVE_STORE_BACKEND"] = args.store_backend
    run_workers(num_workers=args.workers, poll_seconds=args.poll_seconds)
