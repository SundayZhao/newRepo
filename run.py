from transformers import AutoTokenizer, AutoModel,AdamW, get_linear_schedule_with_warmup,AutoConfig,GPT2LMHeadModel,EncoderDecoderModel
from dataloader import MyDataset
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import os
from tqdm import tqdm, trange
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse
from model import seq2editModel
from tqdm import tqdm
from itertools import cycle

def train(seq2editmodel,train_dataloader,examples_num,logger,
    batch_size=64, epochs=20, lr=2e-5,
    warmup_steps=1000,output_dir="./data/model", output_prefix="position",
    save_model_on_epoch=False,gradient_accumulation_steps=4):
    #训练步数
    num_train_optimization_steps=examples_num*epochs//batch_size
    print('save_model_on_epoch:',save_model_on_epoch)
    print('num_train_optimization_steps:',num_train_optimization_steps)
    #设为训练模式
    seq2editmodel.train()

    seq2editmodel=seq2editmodel.cuda()

    device=torch.device("cuda")
    
    word_optimizer = AdamW(seq2editmodel.word_model.parameters(),lr=lr)
    word_scheduler = get_linear_schedule_with_warmup(
        word_optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
    )

    tag_optimizer = AdamW(seq2editmodel.tag_model.parameters(),lr=lr)
    tag_scheduler = get_linear_schedule_with_warmup(
        tag_optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
    )

    position_optimizer = AdamW(seq2editmodel.position_model.parameters(),lr=lr)
    position_scheduler = get_linear_schedule_with_warmup(
        position_optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
    )

    word_loss=torch.tensor([0])
    tag_loss=torch.tensor([0])
    position_loss=torch.tensor([0])

    device=torch.device("cuda")
    
    epoch_loss=0
    epoch_loss_word=0
    epoch_loss_tag=0
    epoch_loss_pos=0
    print('start training...')
    for step in range(num_train_optimization_steps):
        for nonn in [1]:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,word_ids,word_mask,tag_ids,tag_mask,pos_ids,pos_mask = batch

            word_loss,tag_loss,position_loss = seq2editmodel(True,source_ids,source_mask,word_ids,word_mask,tag_ids,tag_mask,pos_ids,pos_mask)
            
            word_loss=word_loss/gradient_accumulation_steps
            epoch_loss += word_loss.item()
            epoch_loss_word+=word_loss.item()
            word_loss.backward(retain_graph=True)
            
            tag_loss=tag_loss/gradient_accumulation_steps
            epoch_loss += tag_loss.item()
            epoch_loss_tag+=tag_loss.item()
            tag_loss.backward(retain_graph=True)
            
            position_loss=position_loss/gradient_accumulation_steps
            if position_loss != None:
              #loss_den=torch.sum(pos_mask)
              #print('loss_den',position_loss.item(),loss_den)
              #position_loss=position_loss/loss_den
              epoch_loss += position_loss.item()
              epoch_loss_pos+=position_loss.item()
              position_loss.backward()
            else:
              print(f"data {idx} error s_list = {len(s_list)}")
              continue
            if step%100==0 and step >0  :
                print(f"epochs {step//(examples_num//batch_size)}, step {step} loss {epoch_loss}")
                logger.add_scalar("train loss", epoch_loss ,global_step=step)
                logger.add_scalar("train word loss", epoch_loss_word ,global_step=step)
                logger.add_scalar("train tag loss", epoch_loss_tag ,global_step=step)
                logger.add_scalar("train position loss", epoch_loss_pos ,global_step=step)
            #print('step',gradient_accumulation_steps)
            #模拟大batch，多个梯度累积清零
            if step % gradient_accumulation_steps==0 and step>0:
                epoch_loss=0
                epoch_loss_word=0
                epoch_loss_tag=0
                epoch_loss_pos=0
            
                #梯度清零
                word_optimizer.step()
                tag_optimizer.step()
                position_optimizer.step()

                word_optimizer.zero_grad()
                tag_optimizer.zero_grad()
                position_optimizer.zero_grad()
                
                
                word_scheduler.step()
                tag_scheduler.step()
                position_scheduler.step()
                
                seq2editmodel.word_model.zero_grad()
                seq2editmodel.tag_model.zero_grad()
                seq2editmodel.position_model.zero_grad()
                
        #每个epoch保存模型
        #(examples_num//batch_size)
        if step%5000==0 and step >0 and save_model_on_epoch :
            print(f"save model epochs {step//(examples_num//batch_size)}, step {step}")
            torch.save(seq2editmodel.state_dict(),os.path.join(output_dir, f"word-all-{step}.pt"),)
    #训练结束，保存模型
    torch.save(seq2editmodel.state_dict(),os.path.join(output_dir, f"word-all-final.pt"),)
    return 


def eval(seq2editmodel,test_data,logger,output_dir="./data/decode"):
  seq2editmodel.eval()
  seq2editmodel=seq2editmodel.cuda()
  device=torch.device("cuda")
  acc_total=0
  acc_word=0
  acc_tag=0
  acc_pos=0

  output_src=open(os.path.join(output_dir, f"output.src.aecm"),'w',encoding='utf-8')
  output_tgt=open(os.path.join(output_dir, f"output.tgt.aecm"),'w',encoding='utf-8')
  output_word=open(os.path.join(output_dir, f"output.word.aecm"),'w',encoding='utf-8')
  output_tag=open(os.path.join(output_dir, f"output.tag.aecm"),'w',encoding='utf-8')
  output_position=open(os.path.join(output_dir, f"output.pos.aecm"),'w',encoding='utf-8')

  encoder_decer_framework=EncoderDecoderModel(None,seq2editmodel.codebertmodel.encoder,seq2editmodel.word_model)
  for idx in tqdm(range(len(test_data))):
    source_input = test_data[idx]
    tag,adding_word,position=seq2editmodel(False,source_input,encoder_decer_framework=encoder_decer_framework)
    print(' generate:',adding_word)
    print(' reference：',source_input[3])
    print('------------------')
    # show all result
    ts=tag[0].split()
    t=[ts[0]]
    p=[position[0]]
    for i in range(1,min(len(position),len(ts))):
        if t[-1]!= ts[i]:
            t.append(ts[i])
            p.append(position[i])
    tag=[' '.join(t)]
    position=p
    if adding_word[0].strip()==source_input[3] and tag[0]==source_input[4] and position==source_input[5]:
      acc_total=acc_total+1
    if adding_word[0].strip()==source_input[3]:
      acc_word=acc_word+1
    if tag[0]==source_input[4]:
      acc_tag=acc_tag+1
    if position==source_input[5]:
      acc_pos=acc_pos+1

    #write result
    output_src.write(source_input[1]+'\n')
    output_word.write(' '.join(adding_word)+'\n')
    output_tag.write(' '.join(tag)+'\n')
    output_position.write(' '.join('%s' %id for id in position)+'\n')
    output_tgt.write(source_input[2]+'\n')

    if idx % 10==0 and idx>0:
      output_src.flush()
      output_word.flush()
      output_tag.flush()
      output_position.flush()
      output_tgt.flush()
    if idx % 50==0 and idx>0:
      print(f"Accuracy :{acc_total/(idx+1)}")
      print(f"Accuracy :{acc_word/(idx+1)}")
      print(f"Accuracy :{acc_tag/(idx+1)}")
      print(f"Accuracy :{acc_pos/(idx+1)}")
  print(f"Accuracy :{acc_total/len(test_data)}")
  print(f"Accuracy :{acc_word/len(test_data)}")
  print(f"Accuracy :{acc_tag/len(test_data)}")
  print(f"Accuracy :{acc_pos/len(test_data)}")
  output_src.close()
  output_word.close()
  output_tag.close()
  output_position.close()
  output_tgt.close()
  print('写出结果成功:'+output_dir)
  return acc_total,acc_word,acc_tag,acc_pos



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--do_train", action='store_true',help="Whether to run training.")
  parser.add_argument("--do_test", action='store_true',help="Whether to run eval on the dev set.")
  parser.add_argument("--load_model_path", type=str,help="load_model_path") 
  parser.add_argument("--pretrain_codebert",default='microsoft/codebert-base', type=str,help="load_model_path") 
  parser.add_argument("--pretrain_codegpt",default='microsoft/CodeGPT-small-java', type=str,help="load_model_path") 
  parser.add_argument("--bs", default=2,type=int,help="batch_size")
  parser.add_argument("--epoch", default=60,type=int,help="epoch")
  parser.add_argument("--max_input_len", default=150,type=int,help="max_input_len")
  parser.add_argument("--max_output_len", default=100,type=int,help="max_output_len")
  parser.add_argument("--load_data_from_cache", default=False,type=bool,help="load_data_from_cache")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=20,help="Number of updates steps to accumulate before performing a backward/update pass.")
  args = parser.parse_args()

  print('--do_train',args.do_train)
  print('--do_test',args.do_test)
  print('--batch_size',args.bs)
  print('--epoch',args.epoch)
  print('--max_input_len',args.max_input_len)
  print('--max_output_len',args.max_output_len)
  print('--gradient_accumulation_steps',args.gradient_accumulation_steps)
  print('--load_model_path',args.load_model_path)
  print('--load_data_from_cache',args.load_data_from_cache)
  if args.do_train:
    #训练
    device=torch.device("cuda")
    seq2editmodel=seq2editModel(args.pretrain_codebert,args.pretrain_codegpt,device)
    seq2editmodel.to(device)
    if args.load_data_from_cache:
      all_source_ids=torch.load('./datasetcache/all_source_ids.pt')
      all_source_mask = torch.load('./datasetcache/all_source_mask.pt')
      all_word_ids = torch.load('./datasetcache/all_word_ids.pt')
      all_word_mask =torch.load('./datasetcache/all_word_mask.pt') 
      all_tag_ids = torch.load('./datasetcache/all_tag_ids.pt')
      all_tag_mask = torch.load('./datasetcache/all_tag_mask.pt')
      all_pos_ids =torch.load('./datasetcache/all_pos_ids.pt')
      all_pos_mask = torch.load('./datasetcache/all_pos_mask.pt')
      print('number of train_examples:',len(all_source_ids))
    else:
      dataset = MyDataset("train")
      train_examples = dataset.read_examples('train')
      examples_num=len(train_examples)
      print('number of train_examples:',examples_num)
      train_features = dataset.convert_examples_to_features(train_examples,args.max_input_len,args.max_output_len)
      all_source_ids = torch.tensor([f[0]['input_ids'] for f in train_features])
      all_source_mask = torch.tensor([f[0]['attention_mask'] for f in train_features])
      all_word_ids = torch.tensor([f[1]['input_ids'] for f in train_features])
      all_word_mask = torch.tensor([f[1]['attention_mask'] for f in train_features]) 
      all_tag_ids = torch.tensor([f[2]['input_ids'] for f in train_features])
      all_tag_mask = torch.tensor([f[2]['attention_mask'] for f in train_features]) 
      all_pos_ids = torch.tensor([f[3]['input_ids'] for f in train_features])
      all_pos_mask = torch.tensor([f[3]['attention_mask'] for f in train_features]) 
      """
      torch.save(all_source_ids,'./datasetcache/all_source_ids.pt')
      torch.save(all_source_mask,'./datasetcache/all_source_mask.pt')
      torch.save(all_word_ids,'./datasetcache/all_word_ids.pt')
      torch.save(all_word_mask,'./datasetcache/all_word_mask.pt') 
      torch.save(all_tag_ids,'./datasetcache/all_tag_ids.pt')
      torch.save(all_tag_mask,'./datasetcache/all_tag_mask.pt')
      torch.save(all_pos_ids,'./datasetcache/all_pos_ids.pt')
      torch.save(all_pos_mask,'./datasetcache/all_pos_mask.pt')
      """
      print('save the dataset tensor to cache')
    
    train_data = TensorDataset(all_source_ids,all_source_mask,all_word_ids,all_word_mask,all_tag_ids,all_tag_mask,all_pos_ids,all_pos_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.bs)
    train_dataloader=cycle(train_dataloader)
    #dataset = MyDataset("train")
    logger = SummaryWriter(log_dir="./data/log2")
    if args.load_model_path is not  None:
        #从指定权重加载，继续训练
        seq2editmodel.load_state_dict(torch.load(args.load_model_path))
    
    train(seq2editmodel,train_dataloader,examples_num,logger,batch_size=args.bs,epochs=args.epoch,save_model_on_epoch=True,gradient_accumulation_steps=args.gradient_accumulation_steps)
  if args.do_test:
    device=torch.device("cuda")
    print('init model...')
    seq2editmodel=seq2editModel(args.pretrain_codebert,args.pretrain_codegpt,device)
    seq2editmodel.to(device)
    if args.load_model_path is None:
        raise ValueError("there is no load_model_path")
    print('load model checkpoint')
    seq2editmodel.load_state_dict(torch.load(args.load_model_path))
    print('load test dataset')
    dataset = MyDataset("test")
    test_examples = dataset.read_examples('test')
    examples_num=len(test_examples)
    print('number of test_examples:',examples_num)
    test_features = dataset.convert_examples_to_features(test_examples,args.max_input_len,args.max_output_len,'test')
    all_test_input = [f for f in test_features]
    """
    all_source_mask = [f[0]['attention_mask'] for f in test_features]
    all_word_ids = [f[1]['input_ids'] for f in test_features]
    all_word_mask = [f[1]['attention_mask'] for f in test_features] 
    all_tag_ids = [f[2]['input_ids'] for f in test_features]
    all_tag_mask = [f[2]['attention_mask'] for f in test_features]
    all_pos_ids = [f[3]['input_ids'] for f in test_features]
    all_pos_mask = [f[3]['attention_mask'] for f in test_features]
    """
    logger = SummaryWriter(log_dir="./data/log2")
    eval(seq2editmodel,all_test_input,logger)

if __name__ == "__main__":
  main()