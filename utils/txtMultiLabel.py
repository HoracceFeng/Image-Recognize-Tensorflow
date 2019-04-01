import os, sys

txtfile  = open(sys.argv[1]).readlines()
outfile  = open(sys.argv[2], 'w')
dictfile = open('dict/Psign24-big.dict').readlines()
big_class = ['pl', 'pr', 'ph', 'pw', 'pa', 'pm', 'pn', 'pd', 'pg', 'po']     ## first 6 are class_w_num, pn='blue'(pn), 'pd'='red'(pne, ps), 'pg', 'po'(others) 

classes = []
for line in dictfile:
	cate = line.strip()
	if cate not in classes:
		classes.append(cate)

for line in txtfile:
	path, cate = line.strip().split('\t')
	if cate[:2] in big_class[:6]:
		outfile.write('{}\t{}\n'.format(path, cate[:2]+','+cate))
	elif cate == 'pn' or cate == 'pg':
		outfile.write('{}\t{}\n'.format(path, cate))
	elif cate in ['ps', 'pne']:
		outfile.write('{}\t{}\n'.format(path, 'pd,'+cate))
	else:
		outfile.write('{}\t{}\n'.format(path, 'po,'+cate))

outfile.close()