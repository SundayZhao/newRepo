import argparse
import os
from tqdm import tqdm
import time
from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple

Match = _namedtuple('Match', 'a b size')
def _calculate_ratio(matches, length):
    if length:
        return 2.0 * matches / length
    return 1.0

class editTagGen:
    def __init__(self, a='', b='', isjunk=None,autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seqs(a.strip().split(), b.strip().split())

    def do_CodeTagGen(self):
        res_tags=[]
        res_addingwords=[]
        res_position=[]
        for tag, i1, i2, j1, j2 in self.get_opcodes():
            if tag == 'delete':
                res_tags.append('delete')
                res_position.append(str(i2))
                res_addingwords.append('self')
            elif tag == 'equal':
                res_tags.append('self')
                res_position.append(str(i2))
                res_addingwords.append('self')
            elif tag == 'insert':
                res_tags.append('insert')
                res_position.append(str(i2))
                res_addingwords.append(' '.join(self.b[j1:j2]))
            elif tag == 'replace':
                res_tags.append('replace')
                res_position.append(str(i2))
                res_addingwords.append(' '.join(self.b[j1:j2]))
        return res_tags,res_addingwords,res_position

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        if a is self.a:
            return
        self.a = a
        self.matching_blocks = self.opcodes = None

    def set_seq2(self, b):
        if b is self.b:
            return
        self.b = b
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self.__chain_b()
    def __chain_b(self):
        b = self.b
        self.b2j = b2j = {}

        for i, elt in enumerate(b):
            indices = b2j.setdefault(elt, [])
            indices.append(i)

        self.bjunk = junk = set()
        isjunk = self.isjunk
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk: # separate loop 
                del b2j[elt]

        self.bpopular = popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idxs in b2j.items():
                if len(idxs) > ntest:
                    popular.add(elt)
            for elt in popular: # ditto; as fast for 1% deletion
                del b2j[elt]

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
        a, b, b2j, isbjunk = self.a, self.b, self.b2j, self.bjunk.__contains__
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        besti, bestj, bestsize = alo, blo, 0

        j2len = {}
        nothing = []
        for i in range(alo, ahi):
            j2lenget = j2len.get
            newj2len = {}
            for j in b2j.get(a[i], nothing):
                # a[i] matches b[j]
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = newj2len[j] = j2lenget(j-1, 0) + 1
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k
            j2len = newj2len

        while besti > alo and bestj > blo and \
              not isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              not isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        while besti > alo and bestj > blo and \
              isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize = bestsize + 1

        return Match(besti, bestj, bestsize)


    def get_matching_blocks(self):
        if self.matching_blocks is not None:
            return self.matching_blocks
        la, lb = len(self.a), len(self.b)
        queue = [(0, la, 0, lb)]
        matching_blocks = []
        while queue:
            alo, ahi, blo, bhi = queue.pop()
            i, j, k = x = self.find_longest_match(alo, ahi, blo, bhi)
            if k:   # if k is 0, there was no matching block
                matching_blocks.append(x)
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i+k < ahi and j+k < bhi:
                    queue.append((i+k, ahi, j+k, bhi))
        matching_blocks.sort()
        i1 = j1 = k1 = 0
        non_adjacent = []
        for i2, j2, k2 in matching_blocks:
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1:
                    non_adjacent.append((i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1:
            non_adjacent.append((i1, j1, k1))

        non_adjacent.append( (la, lb, 0) )
        self.matching_blocks = list(map(Match._make, non_adjacent))
        return self.matching_blocks

    def get_opcodes(self):
        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        self.opcodes = answer = []
        for ai, bj, size in self.get_matching_blocks():
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                answer.append( (tag, i, ai, j, bj) )
            i, j = ai+size, bj+size
            if size:
                answer.append( ('equal', ai, i, bj, j) )
        return answer

    def get_grouped_opcodes(self, n=3):
        codes = self.get_opcodes()
        if not codes:
            codes = [("equal", 0, 1, 0, 1)]
        if codes[0][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = tag, max(i1, i2-n), i2, max(j1, j2-n), j2
        if codes[-1][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = tag, i1, min(i2, i1+n), j1, min(j2, j1+n)

        nn = n + n
        group = []
        for tag, i1, i2, j1, j2 in codes:
            if tag == 'equal' and i2-i1 > nn:
                group.append((tag, i1, min(i2, i1+n), j1, min(j2, j1+n)))
                yield group
                group = []
                i1, j1 = max(i1, i2-n), max(j1, j2-n)
            group.append((tag, i1, i2, j1 ,j2))
        if group and not (len(group)==1 and group[0][0] == 'equal'):
            yield group

    def ratio(self):
        matches = sum(triple[-1] for triple in self.get_matching_blocks())
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self):
        if self.fullbcount is None:
            self.fullbcount = fullbcount = {}
            for elt in self.b:
                fullbcount[elt] = fullbcount.get(elt, 0) + 1
        fullbcount = self.fullbcount
        avail = {}
        availhas, matches = avail.__contains__, 0
        for elt in self.a:
            if availhas(elt):
                numb = avail[elt]
            else:
                numb = fullbcount.get(elt, 0)
            avail[elt] = numb - 1
            if numb > 0:
                matches = matches + 1
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def real_quick_ratio(self):
        la, lb = len(self.a), len(self.b)
        return _calculate_ratio(min(la, lb), la + lb)



parser = argparse.ArgumentParser()
parser.add_argument("type", type=str,choices=['train','dev','test'],help="Whether to run training.")
args = parser.parse_args()

if not os.path.exists(r'.\data'):
  os.makedirs(r'.\data')
if not os.path.exists(r'.\data\{}'.format(args.type)):
  os.makedirs(r'.\data/{}'.format(args.type))
buggycodeFile=open('{}.buggy-fixed.buggy'.format(args.type),'r',encoding='utf-8')
fixedcodeFile=open('{}.buggy-fixed.fixed'.format(args.type),'r',encoding='utf-8')
outputPosFile=open('./data/{}/{}.set.pos'.format(args.type,args.type),'w',encoding='utf-8')
outputTagFile=open('./data/{}/{}.set.tag'.format(args.type,args.type),'w',encoding='utf-8')
outputWordFile=open('./data/{}/{}.set.word'.format(args.type,args.type),'w',encoding='utf-8')
outputSrcFile=open('./data/{}/{}.set.src'.format(args.type,args.type),'w',encoding='utf-8')
outputTgtFile=open('./data/{}/{}.set.tgt'.format(args.type,args.type),'w',encoding='utf-8')
buggycodes=buggycodeFile.readlines()
fixedcodes=fixedcodeFile.readlines()
assert len(buggycodes)==len(fixedcodes)
total_editNum=0
start = time.perf_counter()
for index in tqdm(range(len(buggycodes))):
  buggycode=buggycodes[index].strip()
  fixedcode=fixedcodes[index].strip()
  genTool=editTagGen(buggycode,fixedcode)
  res_tags,res_addingwords,res_position=genTool.do_CodeTagGen()
  total_editNum+=len(res_tags)+2
  assert len(res_tags)==len(res_addingwords)==len(res_position)
  
  outputPosFile.write('<<<<>>>>'.join(res_position)+'\n')
  outputWordFile.write('<<<<>>>>'.join(res_addingwords)+'\n')
  outputTagFile.write('<<<<>>>>'.join(res_tags)+'\n')
  outputSrcFile.write(buggycode+'\n')
  outputTgtFile.write(fixedcode+'\n')
  
end = time.perf_counter()
print('Average number of editing intervals：',total_editNum/len(buggycodes))
print("Total time：", end - start)
buggycodeFile.close()
fixedcodeFile.close()
outputPosFile.close()
outputTagFile.close()
outputWordFile.close()
outputSrcFile.close()
outputTgtFile.close()