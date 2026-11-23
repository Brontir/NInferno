"""merge18.py (lab, 24.11): put gen18.py's T 1-8 bands in front of a routes/pick file's own bands, per problem."""
import json, sys
base, lab = json.load(open(sys.argv[1], encoding='utf-8')), json.load(open(sys.argv[2], encoding='utf-8'))
assert base['gpu']['class'] == lab['gpu']['class'], (base['gpu'], lab['gpu'])
key = lambda e: (e['format'], e['epilogue'], e['n'], e['k'])
have = {key(e): e for e in base['entries']}
for e in lab['entries']:
    if key(e) in have:
        have[key(e)]['bands'] = e['bands'] + [b for b in have[key(e)]['bands'] if b[0] > 8]
    else:
        base['entries'].append(e)
base['source'] += '; lab: T 1-8 bands of tools/route_pick/gen18.py in front'
open(sys.argv[1], 'w', encoding='utf-8', newline='\n').write(json.dumps(base, indent=1) + '\n')
print(sys.argv[1], len(base['entries']))
