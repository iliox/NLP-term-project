DATA_DIR = '../NERdata/NCBI-disease'
f = open('ncbi_test.tsv')
text = f.read().split('\n\n')
f2 = open('ncbi_test.txt','w')
for sent in text:
	lines = sent.split('\n')
	a = []
	b = []
	for j in lines:
		tmp = j.split('\t')
		if(len(tmp) <= 1):
			continue
		a.append(tmp[0])
		b.append(tmp[1])

	if(len(a) > 0):
		f2.write(' '.join(a))
		f2.write('\t')
		f2.write(' '.join(b))
		f2.write('\n')

f.close()
f2.close()

