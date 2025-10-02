#!/usr/bin/env python3
"""
Development script for AI Server.

Provides convenient commands for development, testing, and debugging.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, env=None):
    """Run command and return result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or Path.cwd(),
            env=env or os.environ.copy(),
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def cmd_run(args):
    """Run the server in development mode."""
    print("🚀 Starting AI Server in development mode...")

    # Set development environment variables
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path.cwd() / "src"),
            "AI_SERVER_DEV": "1",
        }
    )

    # Run with uvicorn for auto-reload
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "ai_server.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--reload",
        "--reload-dir",
        "src",
        "--log-level",
        "info",
    ]

    print(f"📍 Server will be available at http://{args.host}:{args.port}")
    print(f"📖 API docs at http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop")

    run_command(" ".join(cmd), env=env)


def cmd_test(args):
    """Run tests."""
    print("🧪 Running tests...")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    if args.unit:
        cmd = "python -m pytest tests/unit/ -v"
    elif args.integration:
        cmd = "python -m pytest tests/integration/ -v"
    elif args.performance:
        cmd = "python -m pytest tests/performance/ -v"
    else:
        cmd = "python -m pytest tests/ -v"

    if args.coverage:
        cmd += " --cov=src/ai_server --cov-report=html --cov-report=term"

    run_command(cmd, env=env)


def cmd_lint(args):
    """Run linting and type checking."""
    print("🔍 Running code quality checks...")

    commands = [
        "python -m py_compile src/ai_server/**/*.py",
        "python -m pyright",
        "python -m ruff check src/ tests/",
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        if not run_command(cmd):
            print(f"❌ Failed: {cmd}")
            return False

    print("✅ All checks passed!")
    return True


def cmd_clean(args):
    """Clean up generated files."""
    print("🧹 Cleaning up...")

    patterns = [
        "*.pyc",
        "__pycache__",
        "*.pyo",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "*.egg-info",
        "dist",
        "build",
        ".mypy_cache",
        ".pyright",
    ]

    for pattern in patterns:
        run_command(f"find . -name '{pattern}' -type f -delete")
        run_command(
            f"find . -name '{pattern}' -type d -exec rm -rf {{}} + 2>/dev/null || true"
        )


def cmd_docs(args):
    """Open documentation."""
    print("📚 Opening documentation...")

    urls = {
        "api": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health/detailed",
        "metrics": "http://localhost:8000/metrics",
    }

    if args.type in urls:
        url = urls[args.type]
        print(f"Opening {url}")
        run_command(
            f"xdg-open {url} 2>/dev/null || open {url} 2>/dev/null || echo 'Please open: {url}'"
        )
    else:
        print("Available docs:")
        for name, url in urls.items():
            print(f"  {name}: {url}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Server Development Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev.py run                    # Start server with auto-reload
  python dev.py run --port 3000        # Start on custom port
  python dev.py test                   # Run all tests
  python dev.py test --unit            # Run only unit tests
  python dev.py test --coverage        # Run tests with coverage
  python dev.py lint                   # Run code quality checks
  python dev.py clean                  # Clean up generated files
  python dev.py docs api               # Open API documentation
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run server in development mode")
    run_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    run_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    run_parser.set_defaults(func=cmd_run)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    test_parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    test_parser.add_argument(
        "--performance", action="store_true", help="Run only performance tests"
    )
    test_parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    test_parser.set_defaults(func=cmd_test)

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Run code quality checks")
    lint_parser.set_defaults(func=cmd_lint)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up generated files")
    clean_parser.set_defaults(func=cmd_clean)

    # Docs command
    docs_parser = subparsers.add_parser("docs", help="Open documentation")
    docs_parser.add_argument(
        "type",
        choices=["api", "health", "metrics"],
        help="Type of documentation to open",
    )
    docs_parser.set_defaults(func=cmd_docs)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Check if we're in the right directory
    if not (Path.cwd() / "src" / "ai_server").exists():
        print("❌ Error: Please run this script from the ai_server root directory")
        sys.exit(1)

    # Run the command
    args.func(args)


if __name__ == "__main__":
    main()
