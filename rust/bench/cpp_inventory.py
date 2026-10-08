#!/usr/bin/env python3
"""The C++ still compiled in a HIGHS_RUST build: every source of libhighs
(unity batches expanded) and the app, with its code lines (no blank or
comment lines) in the pure C++ build and those left under HIGHS_RUST
(`unifdef -DHIGHS_RUST` / `-UHIGHS_RUST`). Prints a markdown table.
  rust/bench/cpp_inventory.py [RUST_BUILD]   (a configured Makefile build)
"""
import os, re, subprocess, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
build = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'build-rust')
tdir = os.path.join(build, 'highs', 'CMakeFiles', 'highs.dir')
objs = set(re.findall(r'^highs/CMakeFiles/highs\.dir/(\S+)\.o:',
                      open(os.path.join(tdir, 'build.make')).read(), re.M))
srcs = []
for o in sorted(objs):
    if o.startswith('Unity/'):
        srcs += re.findall(r'^#include "(.*)"', open(os.path.join(tdir, o)).read(), re.M)
    else:
        srcs.append(os.path.join(root, 'highs', o))
srcs = sorted(os.path.relpath(s, root) for s in srcs) + ['app/RunHighs.cpp']


# What the C++ left in a file does under HIGHS_RUST (by file or directory
# prefix; the first match wins)
NOTES = [
    ('highs/HighsExternal', 'live: third-party notice, extras library loader'),
    ('highs/io/HMpsFF.cpp', 'empty: the free MPS parser is Rust (only the class is used)'),
    ('highs/io/HMPSIO.cpp', 'live: writeModelAsMps (the fixed-format reader is left out)'),
    ('highs/io/FilereaderMps.cpp', 'glue: calls the Rust MPS parser (free format only)'),
    ('highs/io/FilereaderLp.cpp', 'glue: calls the Rust LP reader'),
    ('highs/io/Filereader.cpp', 'live: file type dispatch'),
    ('highs/io/HighsIO.cpp', 'live: highsLogUser/highsLogDev, log callbacks'),
    ('highs/ipm/IpxWrapper.cpp', 'live: IPX glue (options, status, crossover, solution)'),
    ('highs/ipm/ipx/', 'glue: ipx::LpSolver over the Rust IPX'),
    ('highs/lp_data/Highs.cpp', 'live: the Highs API (model passing, getters, IIS/rays drivers)'),
    ('highs/lp_data/HighsInterface.cpp', 'live: Highs internals (model edits, basis, rays, IIS, ill-conditioning)'),
    ('highs/lp_data/HighsIis.cpp', 'live: IIS (not ported)'),
    ('highs/lp_data/HighsCallback.cpp', 'live: user callback data'),
    ('highs/lp_data/HighsDeprecated.cpp', 'API: deprecated wrappers'),
    ('highs/lp_data/HighsLpUtils.cpp', 'live: LP reporting, getSubVectors, deletions, assess primal solution'),
    ('highs/lp_data/HighsModelUtils.cpp', 'live: names, status strings, normaliseNames'),
    ('highs/lp_data/HighsSolution.cpp', 'live: IPX solution handling, unscaling'),
    ('highs/lp_data/HighsRunData.cpp', 'live: run data (record of a run)'),
    ('highs/lp_data/HighsLp.cpp', 'live: HighsLp methods (equality, names, dimensions)'),
    ('highs/lp_data/HighsSolutionDebug.cpp', 'no-op stubs (debugging is left out)'),
    ('highs/lp_data/HighsInfoDebug.cpp', 'no-op stubs (debugging is left out)'),
    ('highs/lp_data/HighsDebug.cpp', 'live: debug status helpers'),
    ('highs/lp_data/', 'part ported (see "The top level")'),
    ('highs/presolve/ICrashX.cpp', 'live: callCrossover (Highs::crossover)'),
    ('highs/presolve/HPresolveAnalysis.cpp', 'live: presolve rule analysis (log)'),
    ('highs/presolve/', 'presolve glue / C++ owner of the postsolve stack'),
    ('highs/mip/', 'MIP (concurrent port): C++ class handles, callbacks, init/restart/workers'),
    ('highs/parallel/', 'task scheduler (concurrent port)'),
    ('highs/model/', 'live: HighsHessian, HighsModel'),
    ('highs/qpsolver/', 'QP glue: phase 1 and instance building'),
    ('highs/simplex/HEkkPrimal.cpp', 'fallback: C++ primal simplex (product form update)'),
    ('highs/simplex/HEkkDual.cpp', 'fallback: C++ dual simplex (product form update); callbacks of the Rust dual'),
    ('highs/simplex/HighsSimplexAnalysis.cpp', 'live: iteration/INVERT logs, timers (analysis stubbed)'),
    ('highs/simplex/HEkkDebug.cpp', 'no-op stubs (debugging is left out)'),
    ('highs/simplex/HSimplexNlaDebug.cpp', 'no-op stubs (debugging is left out)'),
    ('highs/simplex/HSimplexDebug.cpp', 'live: CHUZC failure reports (dev log)'),
    ('highs/simplex/', 'live: HEkk data owner, NLA wrapper, setup, reports'),
    ('highs/util/HFactorDebug.cpp', 'no-op stubs (debugging is left out)'),
    ('highs/util/HighsMatrixPic.cpp', 'debug only (matrix pictures)'),
    ('highs/util/', 'live: utilities (sparse matrix, sort, HSet, vectors)'),
    ('app/', 'main: calls the Rust app (rust/src/lp_data/app.rs)'),
]


def note(path):
    return next((n for p, n in NOTES if path.startswith(p)), '')


def code_lines(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return sum(1 for l in text.splitlines() if l.strip() and not l.strip().startswith('//'))


def lines(path, flag):
    r = subprocess.run(['unifdef', flag, os.path.join(root, path)], capture_output=True, text=True)
    return code_lines(r.stdout)


print('| file | C++ build | HIGHS_RUST | kind | what is left |')
print('|---|---:|---:|---|---|')
tot = [0, 0]
for s in srcs:
    cpp, rs = lines(s, '-UHIGHS_RUST'), lines(s, '-DHIGHS_RUST')
    tot[0] += cpp; tot[1] += rs
    base = os.path.basename(s)
    if re.search(r'Rust|Rs\.|_rs\.', base):
        kind = 'glue'
    elif rs == 0:
        kind = 'empty'
    elif 'stubs' in note(s):
        kind = 'stubs'
    elif rs < cpp:
        kind = 'part ported'
    else:
        kind = 'C++'
    print(f'| {s} | {cpp} | {rs} | {kind} | {note(s) if kind != "glue" else ""} |')
print(f'| **total** ({len(srcs)} files) | {tot[0]} | {tot[1]} | | |')
