"""Compare a pure C++ HiGHS build with a HIGHS_RUST build.

  python3 rust/bench/perf.py CPP_HIGHS RUST_HIGHS [--reps N] [--miplib DIR]

Runs each case alternately with both binaries (single thread), takes the
minimum cycles (`/usr/bin/time -l`), checks that the two solves report the
same path (iterations, nodes, objective, status) and prints a Markdown table
for rust/PERFORMANCE.md. Cycles, not wall time: the machine is shared.
"""
import argparse, math, os, re, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "check" / "instances"
DISPATCH = Path(os.environ.get("DISPATCH_DIR", ROOT / "dispatch_milp_2026_09_30"))

def cases(miplib):
    mip = {"mip_rel_gap": 0.01}
    lp = {"presolve": "off"}
    c = []
    for name in ["air05", "neos17", "nu25-pr12", "neos-911970"]:
        c.append(("MIP", name, miplib / f"{name}.mps.gz", mip))
    c.append(("MIP", "dispatch lambda_080458", DISPATCH / "lambda_20260930_080458_22776806.mps", mip))
    c.append(("MIP", "dispatch 3c1b60d6 root", DISPATCH / "lambda_20260930_142050_3c1b60d6.mps",
              {**mip, "mip_max_nodes": 1}))
    for name in ["25fv47", "80bau3b", "greenbea", "perold", "stair"]:
        c.append(("LP dual simplex", name, CHECK / f"{name}.mps", lp))
        c.append(("LP primal simplex", name, CHECK / f"{name}.mps", {**lp, "simplex_strategy": 4}))
    rel = {"solve_relaxation": "true"}
    for name in ["air04", "rail507", "co-100"]:
        c.append(("LP dual simplex", f"{name} relaxation", miplib / f"{name}.mps.gz", rel))
    c.append(("LP dual simplex", "dispatch 3c1b60d6 relaxation", DISPATCH / "lambda_20260930_142050_3c1b60d6.mps", rel))
    for name in ["greenbea", "80bau3b"]:
        c.append(("IPM (IPX)", name, CHECK / f"{name}.mps", {**lp, "solver": "ipm"}))
    for name in ["rail507", "co-100"]:
        c.append(("IPM (IPX)", f"{name} relaxation", miplib / f"{name}.mps.gz", {**rel, "solver": "ipm"}))
    c.append(("IPM (IPX)", "dispatch 3c1b60d6 relaxation", DISPATCH / "lambda_20260930_142050_3c1b60d6.mps", {**rel, "solver": "ipm"}))
    for name in ["25fv47", "greenbea", "stair"]:
        c.append(("PDLP", name, CHECK / f"{name}.mps", {**lp, "solver": "pdlp", "pdlp_iteration_limit": 20000}))
    read = {"time_limit": 0}
    for name in ["co-100", "neos-5052403-cygnet"]:
        c.append(("Read model (time_limit 0)", f"{name}.mps.gz", miplib / f"{name}.mps.gz", read))
    c.append(("Read model (time_limit 0)", "dispatch 3c1b60d6.mps", DISPATCH / "lambda_20260930_142050_3c1b60d6.mps", read))
    c.append(("Read model (time_limit 0)", "dispatch 3c1b60d6.lp", DISPATCH / "lambda_20260930_142050_3c1b60d6.lp", read))
    return [x for x in c if x[2].exists()]

PATH_RE = re.compile(r"^(Model status.*|.*iterations: .*|Objective value.*|  Nodes .*|  LP iterations .*|  Primal bound .*|  Dual bound .*)$", re.M)

def run(binary, inst, opts):
    with tempfile.NamedTemporaryFile("w", suffix=".opts", delete=False) as f:
        f.write("threads = 1\nrandom_seed = 0\n")
        f.writelines(f"{k} = {v}\n" for k, v in opts.items())
    try:
        r = subprocess.run(["/usr/bin/time", "-l", binary, "--options_file", f.name, str(inst)],
                           capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    cyc = int(re.search(r"(\d+)\s+cycles elapsed", r.stderr).group(1))
    return "\n".join(PATH_RE.findall(r.stdout)), cyc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp"); ap.add_argument("rust")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--miplib", type=Path, default=Path(os.environ.get("MIPLIB_DIR", str(Path.home() / "code" / "miplib"))))
    a = ap.parse_args()
    rows, by_group = [], {}
    for group, name, inst, opts in cases(a.miplib):
        best, paths = {a.cpp: None, a.rust: None}, {}
        for _ in range(a.reps):
            for b in (a.cpp, a.rust):
                p, c = run(b, inst, opts)
                paths.setdefault(b, p)
                best[b] = c if best[b] is None else min(best[b], c)
        ratio = best[a.rust] / best[a.cpp]
        same = paths[a.cpp] == paths[a.rust]
        by_group.setdefault(group, []).append(ratio)
        rows.append(f"| {group} | {name} | {best[a.cpp]/1e9:.2f} | {best[a.rust]/1e9:.2f} | {ratio:.3f} | {'yes' if same else '**NO**'} |")
        print(rows[-1], flush=True)
    print("\n| Group | Case | C++ Gcycles | Rust Gcycles | Rust / C++ | Same path |\n|---|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n| Group | Geomean Rust / C++ |\n|---|---|")
    allr = []
    for g, rs in by_group.items():
        allr += rs
        print(f"| {g} | {math.exp(sum(map(math.log, rs)) / len(rs)):.3f} |")
    print(f"| **All** | **{math.exp(sum(map(math.log, allr)) / len(allr)):.3f}** |")

if __name__ == "__main__":
    main()
