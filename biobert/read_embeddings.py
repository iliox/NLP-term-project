import json
import torch

f = open('embeddings.txt')
text = f.read().split('\n')
a = json.loads(text[1])
print(a.keys())
print(a['words'])
print(len(a['words']))
b = torch.tensor(a['embeddings'])
print(a['tags'])
print(b.shape)
f.close()