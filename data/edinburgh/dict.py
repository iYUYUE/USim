import os, sys

def main(argv):
    
	filepath = sys.argv[1]

	if not os.path.isfile(filepath):
	   print("File path {} does not exist. Exiting...".format(filepath))
	   sys.exit()

	list_of_dicts = []
	with open(filepath) as fp:
		for line in fp:
			tdict = {}
			for p in line.split(','):
				if(len(p) > 1):
					tpair = p.split('-')
					print 'Adding pair', tpair[0], '-', tpair[1], '\n'
					tdict[tpair[0]] = tpair[1]
	   		list_of_dicts.append(tdict)

def str_mapping(str, dic):
    words = text.split("_")

    valid = {}
    for word in words:
        valid[word] = True

    ret = []
    for word in words:
        if not valid[word]:
            continue
        if word in dic:
            ret.append(dic[word])
            continue
        for k, v in dic:
            t = k.split("_")
            if t[0] == word:
                for c in t:
                    valid[c] = False
                ret.append(v);
                continue
        print("no pair in dic")
        return None

    if len(ret) != len(set(ret)):
        return None

    return "_".join(ret)

if __name__ == '__main__' : main(sys.argv)