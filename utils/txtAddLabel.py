import os, sys

txtfile = open(sys.argv[1]).readlines()
outfile = open(sys.argv[2], 'w')

for line in txtfile:
	label = os.path.basename(line.strip())[:-4].split('_')[-1]
	outfile.write('{}\t{}\n'.format(line.strip(), label))
outfile.close()