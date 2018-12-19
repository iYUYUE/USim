from collections import defaultdict
import statistics

path = 'testv4.txt'
k1 = []
f1 = []
k2 = []
fk = []

with open(path, 'r', encoding='utf8') as f:
    f = f.readlines()
    for line in f:
        line = line.strip()
        if line.startswith('File') or line.startswith('fully'):
            continue
        else:
            part = line.split(' ')
            starter = part[0]
            if starter == 'K1':
                k1.append(part[1])
            if starter == 'F1':
                f1.append(part[1])
            if starter == 'K2':
                k2.append(part[1])
            if starter == 'FK':
                fk.append(part[1])

para_set = []
for n in range(len(k1)):
    para_set.append((k1[n], k2[n], fk[n]))

lst = defaultdict(int)
lst_number = defaultdict(int)
for s in para_set:
    lst[(s[0], s[1])] += float(s[2])
    lst_number[(s[0], s[1])] += 1

f1 = open('testv4_avg_mean.txt', 'w')
lst = sorted(lst.items(), key=lambda x:x[1], reverse=True)
for x in lst:
    mean = x[1] / lst_number[x[0]]
    num_lst = []
    for s in para_set:
        if (s[0], s[1]) == x[0]:
            num_lst.append(float(s[2]))
    if len(num_lst) > 1:
        sd = statistics.stdev(num_lst)
    else:
        sd = 0
    f1.write('%s\t%d\t%f\t%f\n' % (x[0], lst_number[x[0]], mean, sd))
f1.close()
