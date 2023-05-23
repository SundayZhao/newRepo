import torch
import torch.nn as nn
import torch
from transformers import AutoTokenizer,AutoModel,AdamW, EncoderDecoderModel,get_linear_schedule_with_warmup,AutoConfig,GPT2LMHeadModel,GPT2Tokenizer
from pointer_network import PointerNetwork
import os

class seq2editModel(nn.Module):
  #参数就是codebert和codegpt的路径
  def __init__(self,codebertPath,codegptPath,device):
    super(seq2editModel, self).__init__()
    
    #device
    self.device=device
    self.criterion = torch.nn.CrossEntropyLoss()
    #codebert
    self.codebertmodel = EncoderDecoderModel.from_encoder_decoder_pretrained('microsoft/codebert-base','microsoft/CodeGPT-small-java')
    #config 
    self.codegptconfig=AutoConfig.from_pretrained(codegptPath)
    self.codegptconfig.add_cross_attention=True
    self.codegptconfig.is_decoder=True
    self.codegptconfig.bos_token_id=0
    self.codegptconfig.eos_token_id=2
    self.codegptconfig_tag=AutoConfig.from_pretrained(codegptPath)
    self.codegptconfig_tag.add_cross_attention=True
    self.codegptconfig_tag.is_decoder=True
    self.codegptconfig_tag.bos_token_id=0
    self.codegptconfig_tag.eos_token_id=2
    #self.codegptconfig.vocab_size=50004
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs
    #GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    self.codegpttokenizer = GPT2Tokenizer.from_pretrained(codegptPath,config=self.codegptconfig)
    self.token = ["<delete>","<replace>","<self>","<insert>"]
    self.codegpttokenizer.add_tokens(self.token)
    assert len(self.codegpttokenizer)==50004
    self.word_model = GPT2LMHeadModel.from_pretrained(codegptPath,config=self.codegptconfig)
    self.word_model.resize_token_embeddings(len(self.codegpttokenizer))
    self.tag_model = GPT2LMHeadModel.from_pretrained(codegptPath,config=self.codegptconfig_tag)
    self.tag_model.resize_token_embeddings(len(self.codegpttokenizer))
    self.position_model = PointerNetwork(100,768)


  def tokenlization(self,line,whole = True):
    if whole:
      return self.codegpttokenizer(line,return_tensors="pt")
    return self.codegpttokenizer(line,is_split_into_words=True,return_tensors="pt")
    

  @torch.no_grad()
  def get_tag(self,embeddings):
    force_words = ["<delete>","<replace>","<self>","<insert>"]
    force_words_ids = self.codegpttokenizer(force_words).input_ids
    force_words_ids=[force_words_ids]
    start = "<s>"
    t_list = self.tokenlization(start)
    t_list["input_ids"]=t_list["input_ids"].to(self.device)
    beam_output = self.tag_model.generate(t_list["input_ids"], 
      max_length=30, 
      num_beams=4, 
      early_stopping=True,
      force_words_ids=force_words_ids,
      num_return_sequences=1,
      pad_token_id=1,
      bos_token_id=0,
      eos_token_id=2,
      encoder_hidden_states=torch.tensor([1,2])
    )
    s = self.codegpttokenizer.batch_decode(beam_output[0], skip_special_tokens=True)
    content = [a for a in s if len(a)>0]
    return content

  @torch.no_grad()
  def get_word(self,embeddings):
    start = "<s>"
    t_list = self.tokenlization(start)
    t_list["input_ids"]=t_list["input_ids"].to(self.device)
    beam_output = self.word_model.generate(t_list["input_ids"],
      max_length=30, 
      num_beams=5, 
      early_stopping=True,
      encoder_hidden_states=torch.tensor([1,2])
    )

    s = self.codegpttokenizer.batch_decode(beam_output[0], skip_special_tokens=True)
    content = [a for a in s if len(a)>0]
    return content

  def get_position(self,encoder_embedding,t_encode):
    ans = []
    t_encode=t_encode.tolist()
    for i in range(len(t_encode[0])):
      t_list = torch.tensor([t_encode[0][:i+1]+[0]*(99-i)])
      assert len(t_list[0])==100
      t_list=t_list.to(self.device)
      decode_result = self.tag_model(input_ids=t_list,attention_mask=t_list.ne(0),encoder_hidden_states=encoder_embedding,output_hidden_states=True)
      decode_hidden = decode_result["hidden_states"][-1]
      decoder_output,logits,loss = self.position_model(encoder_embedding,decode_hidden,torch.tensor([[0]*100]).to(self.device))
      outputs = logits[0][-1]
      outputs = torch.argmax(outputs,dim=-1)
      l = outputs.item()
      ans.append(l)
    return ans
    
    
  def forward(self,is_training,source_ids,source_mask=None,word_ids=None,word_mask=None,tag_ids=None,tag_mask=None,pos_ids=None,pos_mask=None,encoder_decer_framework=None): 
    if is_training:
      #训练模式
      #input = torch.tensor(source_ids)[None,:].to(self.device)
      embeddings=self.codebertmodel.encoder(input_ids=source_ids,attention_mask=source_mask,output_attentions =True,output_hidden_states =True)
      embeddings.last_hidden_state = embeddings.last_hidden_state.to(self.device)
      outputs_word = self.word_model(input_ids=word_ids,attention_mask=word_mask,encoder_hidden_states=embeddings.last_hidden_state,encoder_attention_mask=source_mask,labels=word_ids)
      word_loss = outputs_word[0]
      outputs_tag = self.tag_model( input_ids=tag_ids,attention_mask=tag_mask,encoder_hidden_states=embeddings.last_hidden_state,encoder_attention_mask=source_mask,labels=tag_ids,output_hidden_states=True)
      tag_loss = outputs_tag[0]
      """
      #position 指针网络
      with torch.no_grad():
        decode_result = self.tag_model(input_ids=tag_ids,attention_mask=tag_mask,encoder_hidden_states=embeddings.last_hidden_state,encoder_attention_mask=source_mask,output_hidden_states=True)
      """  
      decode_hidden = outputs_tag["hidden_states"][-1]
      decode_hidden = decode_hidden.to(self.device)
      #s1,s2 = decode_hidden.shape[1],embeddings.last_hidden_state.shape[1]
      pos_ids=pos_ids.to(self.device)
      _,_,position_loss = self.position_model(embeddings.last_hidden_state,decode_hidden,pos_ids)
      """
      target = torch.zeros((pos_ids.size()[0],s1,s2))
      #print('s1,s2',s1,s2)
      #print(pos_ids.size()[0],len(pos_ids),torch.sum(pos_mask))
      #修改为batch
      for batch_index in range(len(pos_ids)):
        for j in range(1,torch.sum(pos_mask[batch_index])-1):
          target[batch_index][j][pos_ids[batch_index][j]] = 1

      target = target.to(self.device)
      #position的loss
        
      position_loss = self.criterion(outputs_pos,target)
      """
      return word_loss,tag_loss,position_loss
    else:
      encoder_decer_framework.encoder=self.codebertmodel.get_encoder()
      encoder_decer_framework.decoder=self.word_model
      attention_mask=source_ids[0].ne(0).to(self.device)
      input_ids=source_ids[0].to(self.device)
      encoder_decer_framework.decoder=self.word_model
      w=encoder_decer_framework.generate(input_ids=input_ids,attention_mask=attention_mask,max_length=20,num_beams=5,repetition_penalty=10.0,temperature=1.0,do_sample=True,top_k=50,top_p=0.95,early_stopping=True,bos_token_id=self.codegpttokenizer.bos_token_id,pad_token_id=self.codegpttokenizer.pad_token_id,eos_token_id =2,num_return_sequences=1)

      encoder_decer_framework.decoder=self.tag_model
      t=encoder_decer_framework.generate(input_ids=input_ids,attention_mask=attention_mask,max_length=len(w[0]),num_beams=5,repetition_penalty=8.0,temperature=8.0,do_sample=True,top_k=3,top_p=0.95,early_stopping=False,bos_token_id=self.codegpttokenizer.bos_token_id,pad_token_id=self.codegpttokenizer.pad_token_id,num_return_sequences=1)
      
      em=self.codebertmodel.encoder(input_ids=input_ids,attention_mask=attention_mask,output_attentions =True,output_hidden_states =True)
      em.last_hidden_state = em.last_hidden_state.to(self.device)
      #o3=position_list = self.get_position(em.last_hidden_state,o2)
      encoder_decer_framework.decoder=self.position_model
      self.position_model.t_cache=t
      self.position_model.tagmodel_cache=self.tag_model
      self.position_model.step=0
      p=encoder_decer_framework.generate(input_ids=input_ids,attention_mask=attention_mask,max_length=len(w[0]),num_beams=1,do_sample=True,top_k=10,top_p=0.95,early_stopping=False,num_return_sequences=1)
      
      w= self.codegpttokenizer.batch_decode(w,skip_special_tokens=True)
      t= self.codegpttokenizer.batch_decode(t,skip_special_tokens=True)
      p=p[0][1:]
      return t,w,p




