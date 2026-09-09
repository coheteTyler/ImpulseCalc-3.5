from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--app", "--serve"):
        from .app import main as app_main
        return app_main(argv[1:])
    if not argv or (argv and argv[0] in ("-h", "--help")):
        # No job path → local app (shareable). Job JSON still accepted as first arg.
        if not argv or argv[0] in ("-h", "--help"):
            p = argparse.ArgumentParser(
                prog="impulsecalc3",
                description="ImpulseCalc3 — local app (default) or JSON cascade job",
            )
            p.add_argument("job", nargs="?", help="JSON job; omit to start the local app on :8765")
            p.add_argument("--app", action="store_true", help="start the local app")
            p.add_argument("--skip-solve", action="store_true")
            p.add_argument("--no-solve", action="store_true")
            p.add_argument("--mesh-only", action="store_true")
            p.add_argument("--post-only", action="store_true")
            p.add_argument("--port", type=int, default=8765)
            if argv and argv[0] in ("-h", "--help"):
                p.print_help()
                return 0
    if not argv:
        from .app import main as app_main
        return app_main([])

    p = argparse.ArgumentParser(prog="impulsecalc3", description="ImpulseCalc3 cascade")
    p.add_argument("job", nargs="?", default=None, help="JSON job")
    p.add_argument("--app", action="store_true")
    p.add_argument("--no-solve", action="store_true")
    p.add_argument("--mesh-only", action="store_true")
    p.add_argument("--post-only", action="store_true")
    p.add_argument("--skip-solve", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    if args.app or not args.job:
        from .app import main as app_main
        extra = ["--port", str(args.port)]
        return app_main(extra)
    from .run import run_job
    r = run_job(
        args.job,
        run_mesh=True,
        skip_solve=args.no_solve or args.mesh_only or args.skip_solve,
        post_only=args.post_only,
        run_solve=not (args.no_solve or args.mesh_only or args.post_only or args.skip_solve),
        run_post=not (args.no_solve or args.mesh_only),
    )
    print(json.dumps({k: r[k] for k in ("app", "name", "success", "predicted", "errors", "t_end_s", "n_chords", "case_dir") if k in r}, indent=2))
    if r.get("forces"):
        print("forces climbing=", r["forces"].get("climbing"), "plateau=", r["forces"].get("plateau"))
        for b in r["forces"].get("blades") or []:
            print(f"  blade{b['blade_index']} Ft={b['Ft_N']:.4g} N  Fd={b['Fd_N']:.4g} N  predicted={r.get('predicted')}")
    return 0 if r.get("success") or args.no_solve or args.skip_solve or args.mesh_only or args.post_only else 1


if __name__ == "__main__":
    sys.exit(main())
