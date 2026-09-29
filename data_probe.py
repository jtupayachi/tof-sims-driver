#!/usr/bin/env python3
"""data_probe.py - dump the internal block tree of an IONTOF file so we can see what it contains.

Usage: python data_probe.py <file> [max_depth=4] [out.txt]
Prints (and saves) one line per block:  name[id]  (size in bytes)  and how many children.
Repeated siblings with the same name (mi[0], mi[1]...) are collapsed.
"""
import sys, warnings
from collections import Counter
from pathlib import Path
from pySPM.Block import Block

warnings.filterwarnings("ignore", message=".*DEPRECATED.*")


def walk(blk, depth, maxd, lines, indent=0, path=""):
    try:
        kids = blk.get_list()
    except Exception as e:
        lines.append("  " * indent + f"<unreadable children: {e}>")
        return
    seen = Counter()
    for k in kids:
        seen[k["name"]] += 1
        if seen[k["name"]] > 2:      # collapse long runs of same-name siblings
            continue
        try:
            child = blk.goto_item(k["name"], k["id"], lazy=True)
            nch = len(child.get_list())
            size = len(child.value) if child.value is not None else 0
        except Exception:
            child, nch, size = None, -1, k.get("blen", 0)
        lines.append(f"{'  ' * indent}{k['name']}[{k['id']}]  {size}B  children={nch}")
        if child is not None and nch > 0 and depth < maxd:
            walk(child, depth + 1, maxd, lines, indent + 1)
    for name, n in seen.items():
        if n > 2:
            lines.append(f"{'  ' * indent}... {name}: {n} entries total")


def probe(path, maxd=4, out=None):
    with open(path, "rb") as f:
        header = f.read(8)
        root = Block(f)
        lines = [f"# {path}", f"# header bytes: {header!r}"]
        walk(root, 1, maxd, lines)
        text = "\n".join(lines)
        if out:
            Path(out).write_text(text)
        return text


if __name__ == "__main__":
    md = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    outf = sys.argv[3] if len(sys.argv) > 3 else None
    print(probe(sys.argv[1], md, outf))
