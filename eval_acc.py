from genTagClass import editTagGen

pos=open('./data/decode/output.pos.aecm',encoding='utf-8')
src=open('./data/decode/output.src.aecm',encoding='utf-8')
tag=open('./data/decode/output.tag.aecm',encoding='utf-8')
tgt=open('./data/decode/output.tgt.aecm',encoding='utf-8')
word=open('./data/decode/output.word.aecm',encoding='utf-8')

pos_items=pos.readlines()
src_items=src.readlines()
tag_items=tag.readlines()
tgt_items=tgt.readlines()
word_items=word.readlines()

acc_p=0
acc_t=0
acc_w=0
acc_all=0
for p,s,t,tt,w in zip(pos_items,src_items,tag_items,tgt_items,word_items):
  genTool=editTagGen(s.strip().lower(),tt.strip().lower())
  res_tags,res_addingwords,res_position=genTool.do_CodeTagGen()
  pos_i=p.split()
  flag=True
  if pos_i==res_position:
    acc_p+=1
  else:
    flag=False
  if w.strip().lower().replace(' ','')==(' '.join(res_addingwords)).replace(' ',''):
    acc_w+=1
  else:
    flag=False
  t=t.strip().lower().split()
  t_nop=[t[0]]
  for i in range(1,len(t)):
    if t_nop[-1]!= t[i]:
      t_nop.append(t[i])
  t_nop=[i.replace('<','').replace('>','') for i in t_nop]

  if t_nop==res_tags:
    acc_t+=1
  else:
    flag=False
  if flag:
    acc_all+=1

print('size of result:',len(pos_items))
print('acc of all:',acc_all/len(pos_items))
print('acc of word:',acc_w/len(pos_items))
print('acc of tag:',acc_t/len(pos_items))
print('acc of position:',acc_p/len(pos_items))
