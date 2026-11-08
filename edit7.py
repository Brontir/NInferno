def sub(p, old, new):
    s = open(p, encoding='utf-8').read()
    assert old in s, (p, old[:50])
    open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new, 1))
sub('src/models/speculation/dflash.h','''    // The rows a full-attention layer reads (the last ones): a YaRN drafter's original context
    // length, the length it was trained on, past which its acceptance falls (#549). 0: all rows.''','''    // The rows every layer of a drafter without a sliding layer reads (the last ones): the context
    // length its YaRN rope was scaled from, shorter than max_positions, which it was trained on
    // and past which its acceptance falls (#549). 0: all rows.''')
s=open('src/convert/dense.cpp',encoding='utf-8').read()
i=s.index('    // The length a YaRN rope was scaled from')
print(s[i:i+400])
