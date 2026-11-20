from private_matching.thread_limits import apply_thread_limits

apply_thread_limits()

from private_matching.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
