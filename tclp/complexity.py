# complexity.py
import argparse
import ast
import fnmatch
import os

IGNORE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
               ".venv", "venv", "env", "build", "dist", "node_modules", ".ipynb_checkpoints"}
IGNORE_GLOBS = ["*_20????????????.py"]  # timestamped snapshots like app_20250828170719.py

def should_ignore(path):
    filename = os.path.basename(path)
    for pat in IGNORE_GLOBS:
        if fnmatch.fnmatch(filename, pat):
            return True
    parts = set(os.path.relpath(path).split(os.sep))
    return any(p in IGNORE_DIRS for p in parts)

def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def loc_of(source):
    return sum(1 for line in source.splitlines() if line.strip() and not line.strip().startswith("#"))

# --- Complexity rules (simple but better than before) ---
def node_complexity(n):
    inc = 0
    if isinstance(n, (ast.If, ast.For, ast.While)):
        inc += 1
    elif isinstance(n, ast.Try):
        # Count excepts + else + finally (common convention)
        inc += len(n.handlers) + (1 if n.orelse else 0) + (1 if n.finalbody else 0)
    elif isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
        # A and B and C => 2 decisions
        inc += len(n.values) - 1
    elif isinstance(n, ast.IfExp):  # ternary
        inc += 1
    # Comprehension filters: [x for x in xs if a if b]
    elif isinstance(n, ast.comprehension):
        inc += len(n.ifs)
    return inc

def function_complexity(fn_node):
    # baseline 1 + decisions within the function
    c = 1
    for n in ast.walk(fn_node):
        c += node_complexity(n)
    return c

def top_level_complexity(tree):
    # Count decisions in module top-level, but DO NOT descend into functions/classes
    c = 0
    def walk(nodes):
        nonlocal c
        for n in nodes:
            c += node_complexity(n)
            # Recurse into control blocks at top-level, but skip defs/classes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.iter_child_nodes(n):
                walk([child])
    walk(tree.body)
    return c

def analyze_file(path):
    src = read_file(path)
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return {"path": path, "error": f"SyntaxError: {e.msg} at {e.lineno}:{e.offset}", "loc": loc_of(src)}
    funcs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append({
                "name": n.name,
                "lineno": getattr(n, "lineno", None),
                "complexity": function_complexity(n)
            })
    file_top = top_level_complexity(tree)
    file_total = file_top + sum(f["complexity"] for f in funcs)
    return {
        "path": path,
        "loc": loc_of(src),
        "file_top": file_top,
        "functions": sorted(funcs, key=lambda x: x["complexity"], reverse=True),
        "total": file_total
    }

def main(root, threshold, topn):
    results, errors = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored dirs in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            if should_ignore(path):
                continue
            res = analyze_file(path)
            if "error" in res:
                errors.append(res)
            else:
                results.append(res)

    # Print errors (non-fatal)
    for e in errors:
        print(f"SKIP {e['path']}: {e['error']}")

    # Summarize worst files
    scored = []
    for r in results:
        density = (100.0 * r["total"] / r["loc"]) if r["loc"] else 0.0
        scored.append((r["total"], density, r))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    print("\n=== Worst files by total complexity ===")
    for total, density, r in scored[:topn]:
        print(f"{r['path']}: total={total}, top_level={r['file_top']}, loc={r['loc']}, density={density:.1f}/100LOC")

    # Per-function offenders across the codebase
    func_rows = []
    for _, _, r in scored:
        for f in r["functions"]:
            func_rows.append((f["complexity"], r["path"], f["name"], f["lineno"]))
    func_rows.sort(reverse=True)

    print("\n=== Worst functions across codebase ===")
    for c, path, name, lineno in func_rows[:topn]:
        print(f"{path}:{lineno}  {name}()  complexity={c}")

    # Threshold warnings
    print(f"\n=== Functions >= {threshold} ===")
    for c, path, name, lineno in func_rows:
        if c >= threshold:
            print(f"{path}:{lineno}  {name}()  complexity={c}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    main(args.root, args.threshold, args.top)
