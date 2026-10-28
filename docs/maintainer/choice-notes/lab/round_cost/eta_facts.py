"""kSmallTEta against two fact hypotheses (DECISIONS 54): compute at the dp4a rate, or a fixed time a round.

    python eta_facts.py
"""
import sys, collections, math
sys.argv=['x']
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import fit_eta as F
rows=[r for r in F.rows_of_registry() if F.fitted(r)]
by=collections.defaultdict(list)
for r in rows: by[r['board']['name']].append(r)
print("card | n | eta(own fit) | 4*eta (dp4a share) | BW/SMclk | fixed ms/round if kappa=0 (mean,cv)")
for name,rs in sorted(by.items()):
    b=rs[0]['board']; eta=F.fit(rs)
    sms_clk=b['peak']/(512 if b['gen']=='volta' else 2048)
    fixed=[]
    for r in rs:
        rb=F.R(r['p']['round_facts'],r['mode'],r['k'],0.0)
        tplain=r['p']['round_facts']['target']['bytes']/b['bw']
        fixed.append((r['meas']-rb)*tplain*1e3)
    m=sum(fixed)/len(fixed); sd=(sum((x-m)**2 for x in fixed)/len(fixed))**.5
    share = eta*(1 if b['gen']=='volta' else 4)
    print(f"{name} | {len(rs)} | {eta:.3f} | {share:.2f} | {b['bw']/sms_clk:.2f} | {m:.2f} ({sd/m:.0%})")
