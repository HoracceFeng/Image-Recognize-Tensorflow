import os

dictfile = open(sys.argv[1]).readlines()
classes = []
for line in dictfile:
    classes.append(line.strip())


listfile = open(sys.argv[2]).readlines()
for name in listfile:
    label = os.path.basename(name.strip())[:-4].split('_')[-1]
    if label not in classes:
        continue
    print name.strip()

def counter(listfile, labels):
    for label in labels:
        count = 0
        for name in listfile:
            # _label = os.path.basename(name.strip())[:-4].split('_')[-1]
            _, _label = name.strip().split('\t')
            if _label == label:
                count += 1
        print label, count