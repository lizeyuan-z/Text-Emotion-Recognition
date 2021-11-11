import re

label = []
data = []
with open('data/ISEAR.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('|')
        if tmp[36] not in label:
            label.append(tmp[36])
        data.append([tmp[40], label.index(tmp[36])])

with open('data/ISEAR_processed.txt', 'w', encoding='utf-8') as f:
    for line in data:
        f.write('{0}\t{1}\n'.format(line[0], line[1]))


from collections import Counter
result = Counter(label)
print(result)

label0 = []
data0 = []
lenth = 0
with open('data/2012.1-2012.4', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        tmp1 = re.sub(r'\S*:', '', tmp[1]).split()
        tmp1 = list(map(int, tmp1))[1:]
        if tmp1.index(max(tmp1)) not in label0:
            label0.append(tmp1.index(max(tmp1)))
        data0.append([re.sub(r'\s*', '', tmp[2]), label0.index(tmp1.index(max(tmp1)))])
        lenth += len(re.sub(r'\s*', '', tmp[2]))
label1 = []
data1 = []
transfer = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
with open('data/2016.1-2016.11', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        tmp1 = re.sub(r'\S*:', '', tmp[1]).split()
        tmp1 = list(map(int, tmp1))[1:]
        if tmp1.index(max(tmp1)) not in label1:
            label1.append(tmp1.index(max(tmp1)))
        data1.append([re.sub(r'\s*', '', tmp[2]), transfer[label1.index(tmp1.index(max(tmp1)))]])
        lenth += len(re.sub(r'\s*', '', tmp[2]))

with open('data/SinaNews_processed.txt', 'w', encoding='utf-8') as f:
    for line in data0:
        f.write('{0}\t{1}\n'.format(line[0], line[1]))
    for line in data1:
        f.write('{0}\t{1}\n'.format(line[0], line[1]))
print(len(label), len(label0), len(label1))
print(lenth / (len(data0) + len(data1)))
