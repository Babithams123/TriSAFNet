"""
run_all.py -- Master script for the complete LULC experiment pipeline.
Runs data preparation, all experiments (with optional ablation), map
generation for every requested patch size, and produces a summary report.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --years 2018        # single year only
    python scripts/run_all.py --skip-data         # skip data pipeline (reuse existing)
    python scripts/run_all.py --skip-ablation     # skip ablation study
    python scripts/run_all.py --skip-maps         # skip map generation
    python scripts/run_all.py --no-sleep          # don't sleep/logoff at end
    python scripts/run_all.py --no-mlflow         # disable MLflow tracking
    python scripts/run_all.py --patch-sizes 7 15  # only 7x7 and 15x15
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import RUNS_DIR, DATA_DIR, PATCH_SIZES, PATCH_SIZE

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cleanup_temp_files(logger=None):
    """Remove __pycache__ dirs, .pyc files, and stale checkpoint files."""
    removed = 0
    for cache_dir in PROJECT_ROOT.rglob('__pycache__'):
        shutil.rmtree(cache_dir, ignore_errors=True)
        removed += 1
    for pyc in PROJECT_ROOT.rglob('*.pyc'):
        pyc.unlink(missing_ok=True)
        removed += 1
    stale_ckpt = Path(tempfile.gettempdir()) / 'trisafnet_best_weights.h5'
    if stale_ckpt.exists():
        stale_ckpt.unlink(missing_ok=True)
        removed += 1
    if logger:
        logger.info(f"  Cleanup: removed {removed} temp items")
    return removed


def setup_master_logging():
    """Configure master-level logging to runs/master_run.log."""
    runs_dir = Path(RUNS_DIR)
    runs_dir.mkdir(exist_ok=True, parents=True)
    log_path = runs_dir / 'master_run.log'

    root = logging.getLogger('master')
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(str(log_path), mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return root


def run_stage(logger, name, cmd):
    """Run a pipeline stage as a subprocess, stream output, return success bool."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE: {name}")
    logger.info(f"CMD:   {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        elapsed = time.time() - t0
        logger.info(f"STAGE COMPLETE: {name} ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        logger.error(f"STAGE FAILED: {name} ({elapsed:.1f}s) — exit code {e.returncode}")
        return False
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"STAGE ERROR: {name} ({elapsed:.1f}s) — {e}")
        return False


def run_stages_parallel(logger, stages):
    """Run multiple pipeline stages as parallel subprocesses.
    Falls back to sequential if only one stage."""
    if not stages:
        return {}
    if len(stages) == 1:
        name, cmd = stages[0]
        return {name: run_stage(logger, name, cmd)}

    results = {}
    procs = {}
    logger.info(f"\n  >> Launching {len(stages)} stages in PARALLEL <<")

    for name, cmd in stages:
        logger.info(f"  [PARALLEL] START: {name}")
        logger.info(f"    CMD: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        procs[name] = (proc, time.time())

    for name, (proc, t0) in procs.items():
        proc.wait()
        elapsed = time.time() - t0
        success = proc.returncode == 0
        status = 'COMPLETE' if success else 'FAILED'
        logger.info(f"  [PARALLEL] {status}: {name} ({elapsed:.1f}s)")
        results[name] = success

    return results


def main():
    parser = argparse.ArgumentParser(description='Master pipeline orchestrator')
    parser.add_argument('--years', nargs='+', type=int, default=[2018, 2024],
                        choices=[2018, 2024], help='Years to process')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip TriSAFNet ablation study')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data pipeline (use existing patches)')
    parser.add_argument('--skip-maps', action='store_true',
                        help='Skip map generation')
    parser.add_argument('--no-sleep', action='store_true',
                        help='Do not sleep/logoff after completion')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking in child scripts')
    parser.add_argument('--patch-sizes', nargs='+', type=int,
                        default=PATCH_SIZES,
                        help=f'Patch sizes to test (default: {PATCH_SIZES})')
    args = parser.parse_args()

    logger = setup_master_logging()
    cleanup_temp_files(logger)
    logger.info(f"\n{'#'*60}")
    logger.info(f"  MASTER PIPELINE START — {datetime.now().isoformat()}")
    logger.info(f"  Years:         {args.years}")
    logger.info(f"  Patch sizes:   {args.patch_sizes}")
    logger.info(f"  Skip data:     {args.skip_data}")
    logger.info(f"  Skip ablation: {args.skip_ablation}")
    logger.info(f"  Skip maps:     {args.skip_maps}")
    logger.info(f"  MLflow:        {'disabled' if args.no_mlflow else 'enabled'}")
    logger.info(f"{'#'*60}")

    t_start = time.time()
    stage_results = {}
    scripts_dir = Path(__file__).resolve().parent
    year_arg = 'both' if len(args.years) == 2 else str(args.years[0])

    for ps in args.patch_sizes:
        ps_tag = f"[{ps}x{ps}]"
        suffix = f'_ps{ps}' if ps != PATCH_SIZE else ''

        logger.info(f"\n{'#'*60}")
        logger.info(f"  PATCH SIZE: {ps}x{ps}")
        logger.info(f"{'#'*60}")

        # ---- STAGE 1: DATA PIPELINE (with augmentation) ----
        for year in args.years:
            npz_path = Path(DATA_DIR) / f"patches_{year}_{ps}x{ps}.npz"
            name = f"{ps_tag} Data Pipeline {year}"
            if args.skip_data or npz_path.exists():
                logger.info(f"  SKIP {name}: {'--skip-data' if args.skip_data else npz_path.name + ' already exists'}")
                stage_results[name] = True
            else:
                stage_results[name] = run_stage(logger, name, [
                    PYTHON, str(scripts_dir / '02_data_pipeline.py'),
                    '--year', str(year), '--patch-size', str(ps)
                ])

        # ---- STAGE 2: DATA PIPELINE (no physics aug — needed for ablation) ----
        if not args.skip_ablation:
            for year in args.years:
                noaug_path = Path(DATA_DIR) / f"patches_{year}_{ps}x{ps}_noaug.npz"
                name = f"{ps_tag} Data Pipeline {year} (no-aug)"
                if args.skip_data or noaug_path.exists():
                    logger.info(f"  SKIP {name}: {'--skip-data' if args.skip_data else noaug_path.name + ' already exists'}")
                    stage_results[name] = True
                else:
                    stage_results[name] = run_stage(logger, name, [
                        PYTHON, str(scripts_dir / '02_data_pipeline.py'),
                        '--year', str(year), '--patch-size', str(ps),
                        '--no-physics-aug'
                    ])

        # ---- STAGE 3: EXPERIMENTS + OPTIONAL ABLATION (single invocation) ----
        exp_cmd = [
            PYTHON, str(scripts_dir / '03_run_experiments.py'),
            '--year', year_arg, '--patch-size', str(ps),
            '--suffix', suffix,
        ]
        if not args.skip_ablation:
            exp_cmd.append('--ablation-trisafnet')
        if args.no_mlflow:
            exp_cmd.append('--no-mlflow')

        name = f"{ps_tag} Experiments{' + Ablation' if not args.skip_ablation else ''} ({year_arg})"
        success = run_stage(logger, name, exp_cmd)
        stage_results[name] = success

        # ---- STAGE 4: MAP GENERATION ----
        if not args.skip_maps:
            map_cmd = [
                PYTHON, str(scripts_dir / '04_generate_maps.py'),
                '--patch-size', str(ps),
            ]
            if args.no_mlflow:
                map_cmd.append('--no-mlflow')

            name = f"{ps_tag} Map Generation"
            success = run_stage(logger, name, map_cmd)
            stage_results[name] = success

        cleanup_temp_files(logger)

    # ---- SUMMARY ----
    total_time = time.time() - t_start
    logger.info(f"\n{'#'*60}")
    logger.info(f"  MASTER PIPELINE COMPLETE")
    logger.info(f"  Total elapsed: {total_time/3600:.1f} hours ({total_time:.0f}s)")
    logger.info(f"{'#'*60}")

    logger.info(f"\n  Stage Results:")
    all_ok = True
    for stage_name, success in stage_results.items():
        status = "OK" if success else "FAILED"
        if not success:
            all_ok = False
        logger.info(f"    [{status}] {stage_name}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': round(total_time, 1),
        'years': args.years,
        'patch_sizes': args.patch_sizes,
        'stages': {k: v for k, v in stage_results.items()},
        'all_stages_passed': all_ok,
    }
    summary_path = Path(RUNS_DIR) / 'paper_results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Summary saved to: {summary_path}")

    cleanup_temp_files(logger)

    if not all_ok:
        logger.warning("\n  Some stages FAILED. Check logs above for details.")

    if not args.no_sleep:
        logger.info("\n  All done. Putting system to SLEEP in 30 seconds...")
        logger.info("  Press Ctrl+C to cancel.")
        try:
            time.sleep(30)
            logger.info("  Initiating sleep + logoff...")
            import platform
            if platform.system() == 'Windows':
                subprocess.run(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0', '1', '0'],
                               check=False)
                subprocess.run(['logoff'], shell=True, check=False)
            else:
                subprocess.run(['systemctl', 'suspend'], check=False)
        except KeyboardInterrupt:
            logger.info("  Sleep cancelled by user.")

    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
