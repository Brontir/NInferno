"""One line per plan: the speculation served, its basis and the speedups without and with thinking."""
import json, sys

d = json.load(sys.stdin)
s = d["plan"].get("speculation") or {}
t = s.get("by_thinking") or {}
nt = (t.get("non_thinking") or {}).get("speedup", 0)
th = (t.get("thinking") or {}).get("speedup", 0)
print(f"{sys.argv[1]:34s} {sys.argv[2]:7s} {sys.argv[3]:22s} {s.get('served')} | {str(s.get('basis'))[:60]} | x{nt:.2f} / x{th:.2f}")
