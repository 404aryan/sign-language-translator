import json
d = json.load(open('data/preset-references.json'))
print(f'Words loaded: {len(d)}')
print('Sample words:', list(d.keys())[:10])
first_word = list(d.keys())[0]
print(f'First word ({first_word}) takes: {len(d[first_word])}')
print(f'First take meta: {d[first_word][0]["meta"]}')
print(f'First take frames: {len(d[first_word][0]["frames"])}')
