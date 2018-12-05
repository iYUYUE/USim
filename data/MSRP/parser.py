#!/usr/bin/env python

import codecs, os, sys, re, csv
from nltk.tokenize import word_tokenize

# print the gold alignments for both partitions
def main(argv):

    outf_s_0 = open('msr_source_0.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw_s_0 = streamWriter(outf_s_0)
    
    outf_r_0 = open('msr_reference_0.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw_r_0 = streamWriter(outf_r_0)

    outf_s_1 = open('msr_source_1.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw_s_1 = streamWriter(outf_s_1)
    
    outf_r_1 = open('msr_reference_1.txt', 'wb')
    streamWriter = codecs.lookup("utf-8")[-1]
    outw_r_1 = streamWriter(outf_r_1)

    csvfile = 'msr_paraphrase_train.txt'

    with open(csvfile, newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter='\t', quotechar='|'):
            if row[0] == '0':
                outw_s_0.write(' '.join(word_tokenize(row[3])) + '\n')
                outw_r_0.write(' '.join(word_tokenize(row[4])) + '\n')
            elif row[0] == '1':
                outw_s_1.write(' '.join(word_tokenize(row[3])) + '\n')
                outw_r_1.write(' '.join(word_tokenize(row[4])) + '\n')

    csvfile = 'msr_paraphrase_test.txt'

    with open(csvfile, newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter='\t', quotechar='|'):
            if row[0] == '0':
                outw_s_0.write(' '.join(word_tokenize(row[3])) + '\n')
                outw_r_0.write(' '.join(word_tokenize(row[4])) + '\n')
            elif row[0] == '1':
                outw_s_1.write(' '.join(word_tokenize(row[3])) + '\n')
                outw_r_1.write(' '.join(word_tokenize(row[4])) + '\n')
    
    outf_s_0.flush()
    outf_r_0.flush()
    outf_s_1.flush()
    outf_r_1.flush()


if __name__ == '__main__' : main(sys.argv)