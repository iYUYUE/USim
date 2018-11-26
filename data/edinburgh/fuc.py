import sys

def parseString(str, dic):
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







    


    # 

    for k, v in dic:
        temps = k.split("_")
        for temp in temps:
            # hasRep[]
            if idex[temp] :
