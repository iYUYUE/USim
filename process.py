from collections import defaultdict

# path = 'output\\p4p.corpus.out'
path = 'new_output\\MSRPA.corpus.out'
para_type = defaultdict(int)
para_ucca = {}
total = 0
stop_para_list = ['identical', 'non_paraphrases', 'entailment']

def merge_tok(x):
    if '_' in x:
        x_split = x.split('_')
        return x_split[0] + '_' + x_split[-1]
    else:
        return x

with open(path, 'r', encoding='utf8') as f:
    f = f.readlines()
    line_count = 0
    for line in f:
        p = line.strip().split(',')
        for pair in p[:-1]:
            align = pair.split('-')
            source = merge_tok(align[0])
            target = merge_tok(align[1])
            if '_' not in source and '_' not in target:
                total += 1
                source_ucca_node = align[3]
                target_ucca_node = align[4]
                if source_ucca_node != target_ucca_node:
                    if source_ucca_node == 'H' or target_ucca_node == 'H':
                        a=1
                    else:
                        if align[2] not in stop_para_list:
                            if align[2] not in para_ucca.keys():
                                para_ucca[align[2]] = defaultdict(int)
                            if source_ucca_node != '' and target_ucca_node != '':
                                para_type[align[2]] += 1
                                para_ucca[align[2]][(source_ucca_node, target_ucca_node)] += 1

        line_count += 1

print('total\t%d\n' % total)
t_v = 0
for k, v in para_type.items():
    print('%s = %d' % (k, v))
    t_v += v
print('total paraphrase with same label = %d' % t_v)
print()

for k,dic in para_ucca.items():
    print(k)
    val = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    for item in val:
        print('%s\t%s' % (item[0], item[1]))
    print('\n')
