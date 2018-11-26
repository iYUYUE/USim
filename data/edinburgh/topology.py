#!/usr/bin/env python

# This software prints out the gold alignments in the retokenized
# Edinburgh paraphrase corpus in a human-readable form, demonstrating
# how to read the alignments from the JSON files

# usage: $ ./print_gold.py
#    or: $ python print_gold.py

# Copyright (c) Michael White, 2013

# This software is licensed under the Apache 2.0 license, a copy of which
# is in the same directory as this file


import codecs, os, sys, json, re
from collections import defaultdict


# convert source index to target index list mappings to an
# int to sorted int list dict, assuming a defaultdict(list)
def add_mappings(mappings, alignment_dict):
    for mapping in mappings:
        src_index = mapping[0]
        for targ_index in mapping[1]:
            alignment_dict[src_index].append(targ_index)
            alignment_dict[src_index].sort()

# get atomic phrase pairs as a list of int tuples
def get_atomic_phrase_pairs(alignment_dict):
    # make inverse target tuple to source list dict
    inv_dict = defaultdict(list)
    for (isrc,targlist) in alignment_dict.items():
        targtup = tuple(targlist)
        inv_dict[targtup].append(isrc)
        inv_dict[targtup].sort()
    # return list in src, target order
    retval = [(tuple(srclist),targtup) for (targtup,srclist) in inv_dict.items()]
    retval.sort()
    return retval

# get tokens, with '..' for discontinuities
def get_tokens(tup, tokens):
    retlist = [x+1 for x in tup]
    return '_'.join(str(e) for e in retlist)

    
# print the gold alignments for the given paraphrases and partition
# according to the sure_only option
def print_gold_option(paras, sure_only=True):

    option = 'sure' if sure_only else 'sure+poss'
    
    outfn = 'source.' + option + '.txt'
    outf = open(outfn, 'w')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw = streamWriter(outf)

    print 'Writing source', option, 'to', outfn, '...'

    for pair in paras:

        # print paraphrase pair
        # outw.write('##### ' + pair['id'] + ' #####\n\n')
        src, targ = pair['S']['string'], pair['T']['string']
        # src_toks, targ_toks = src.split(), targ.split()

        outw.write(src + '\n')
        # tmplist = src.split()
        # for idx, val in enumerate(tmplist):
        #     if re.search('[0-9a-zA-Z]', val):
        #         tmplist[idx] = re.sub('[^0-9a-zA-Z]+', '', val)

        # outw.write(' '.join(tmplist) + '\n')

        # # get chosen alignments as atomic phrase pairs
        # anns = pair['annotations']
        # gold = anns[pair['train']]
        # adict = defaultdict(list)
        # gold_sure = gold['S']
        # add_mappings(gold_sure, adict)
        # if not sure_only:
        #     gold_poss = gold['P']
        #     add_mappings(gold_poss, adict)
        # atom_phr_pairs = get_atomic_phrase_pairs(adict)

        # # print atomic phrase pairs
        # for (srctup, targtup) in atom_phr_pairs:
        #     outw.write(get_tokens(srctup, src_toks) + ' <-> ' +
        #                get_tokens(targtup, targ_toks) + '\n')
        # outw.write('\n\n')
    # outw.write('Lodged with me useless , though my soul more bent .')
    outf.flush()

    outfn = 'reference.' + option + '.txt'
    outf = open(outfn, 'w')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw = streamWriter(outf)

    print 'Writing reference', option, 'to', outfn, '...'

    for pair in paras:

        # print paraphrase pair
        # outw.write('##### ' + pair['id'] + ' #####\n\n')
        src, targ = pair['S']['string'], pair['T']['string']
        # src_toks, targ_toks = src.split(), targ.split()
        # outw.write(src + '\n\n')

        outw.write(targ + '\n')
        # tmplist = targ.split()
        # for idx, val in enumerate(tmplist):
        #     if re.search('[0-9a-zA-Z]', val):
        #         tmplist[idx] = re.sub('[^0-9a-zA-Z]+', '', val)

        # outw.write(' '.join(tmplist) + '\n')

        # # get chosen alignments as atomic phrase pairs
        # anns = pair['annotations']
        # gold = anns[pair['train']]
        # adict = defaultdict(list)
        # gold_sure = gold['S']
        # add_mappings(gold_sure, adict)
        # if not sure_only:
        #     gold_poss = gold['P']
        #     add_mappings(gold_poss, adict)
        # atom_phr_pairs = get_atomic_phrase_pairs(adict)

        # # print atomic phrase pairs
        # for (srctup, targtup) in atom_phr_pairs:
        #     outw.write(get_tokens(srctup, src_toks) + ' <-> ' +
        #                get_tokens(targtup, targ_toks) + '\n')
        # outw.write('\n\n')
    # outw.write('Lodged with me useless , though my soul more bent .')
    outf.flush()

    outfn = 'alignment.' + option + '.txt'
    outf = open(outfn, 'w')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw = streamWriter(outf)

    print 'Writing alignment', option, 'to', outfn, '...'

    for pair in paras:

        src, targ = pair['S']['string'], pair['T']['string']
        src_toks, targ_toks = src.split(), targ.split()

        # get chosen alignments as atomic phrase pairs
        anns = pair['annotations']
        gold = anns[pair['train']]
        adict = defaultdict(list)
        gold_sure = gold['S']
        add_mappings(gold_sure, adict)
        if not sure_only:
            gold_poss = gold['P']
            add_mappings(gold_poss, adict)
        atom_phr_pairs = get_atomic_phrase_pairs(adict)

        # print atomic phrase pairs
        for (srctup, targtup) in atom_phr_pairs:
            outw.write(get_tokens(srctup, src_toks) + '-' +
                       get_tokens(targtup, targ_toks) + ',')
        outw.write('\n')

    outf.flush()
    print 'Done.'
    print
    

# print the gold alignments for the given partition
def print_gold(partition1, partition2):

    fn = partition1 + '.json'
    print
    print 'Reading data from', fn, '...'
    json_file = open(fn, 'r')
    json_data = json.load(json_file)
    json_file.close()
    print 'Done.'

    paras1 = json_data['paraphrases']
    print 'Read', len(paras1), 'paraphrase pairs.'
    print

    fn = partition2 + '.json'
    print
    print 'Reading data from', fn, '...'
    json_file = open(fn, 'r')
    json_data = json.load(json_file)
    json_file.close()
    print 'Done.'

    paras2 = json_data['paraphrases']
    print 'Read', len(paras2), 'paraphrase pairs.'
    print

    print_gold_option(paras1 + paras2, sure_only=True)
    # print_gold_option(paras, partition, sure_only=False)


# print the gold alignments for both partitions
def main(argv):
    
    if len(argv[1:]) > 0:
        print 'Ignoring extra args:', argv[1:]

    print_gold('train', 'test')


if __name__ == '__main__' : main(sys.argv)
