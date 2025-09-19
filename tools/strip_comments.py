import os
import io
import json
import argparse
import tokenize

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "venv"}


def strip_py(source: str, remove_docstrings: bool = False) -> str:
    out = []
    last_lineno = -1
    last_col = 0
    prev_toktype = None
    indents = []
    for toktype, ttext, (slineno, scol), (elineno, ecol), line in tokenize.generate_tokens(io.StringIO(source).readline):
        if toktype == tokenize.COMMENT:
            continue
        if remove_docstrings and toktype == tokenize.STRING and prev_toktype in {tokenize.INDENT, tokenize.NEWLINE, None}:                      
            if scol == 0 or (out and out[-1].strip().endswith(':')):
                prev_toktype = toktype
                continue
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            out.append(' ' * (scol - last_col))
        out.append(ttext)
        last_col = ecol
        last_lineno = elineno
        prev_toktype = toktype
    return ''.join(out)


def process_py(path: str, dry_run: bool = False, no_backup: bool = False, remove_docstrings: bool = False) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    stripped = strip_py(src, remove_docstrings=remove_docstrings)
    if stripped != src:
        if dry_run:
            return True
        if not no_backup and not os.path.exists(path + ".bak"):
            with open(path + ".bak", "w", encoding="utf-8") as b:
                b.write(src)
        with open(path, "w", encoding="utf-8") as f:
            f.write(stripped)
        return True
    return False


def process_ipynb(path: str, dry_run: bool = False, no_backup: bool = False, remove_docstrings: bool = False) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src_text = "".join(src) if isinstance(src, list) else src
        stripped = strip_py(src_text, remove_docstrings=remove_docstrings)
        if stripped != src_text:
            changed = True
            cell["source"] = list(stripped.splitlines(keepends=True))
    if changed:
        if dry_run:
            return True
        if not no_backup and not os.path.exists(path + ".bak"):
            with open(path + ".bak", "w", encoding="utf-8") as b:
                json.dump(nb, b, ensure_ascii=False, indent=1)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--remove-docstrings", action="store_true", help="Also remove docstrings (triple-quoted strings at definition starts)")
    args = p.parse_args()

    total = 0
    for dirpath, dirnames, filenames in os.walk(args.root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if name.endswith(".py"):
                if process_py(path, dry_run=args.dry_run, no_backup=args.no_backup, remove_docstrings=args.remove_docstrings):
                    total += 1
            elif name.endswith(".ipynb"):
                if process_ipynb(path, dry_run=args.dry_run, no_backup=args.no_backup, remove_docstrings=args.remove_docstrings):
                    total += 1
    print(f"Modified files: {total}")


if __name__ == "__main__":
    main()
