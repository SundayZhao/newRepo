from transformers import AutoTokenizer, AutoModel,AdamW, get_linear_schedule_with_warmup,AutoConfig,GPT2LMHeadModel,GPT2Tokenizer
import torch
import sys
import os
from torch.utils.data import Dataset, DataLoader

dev = ["./data/dev/dev.set.src","./data/dev/dev.set.word","./data/dev/dev.set.tag","./data/dev/dev.set.pos","./data/dev/dev.set.tgt"]
test = ["./data/test/test.set.src","./data/test/test.set.word","./data/test/test.set.tag","./data/test/test.set.pos","./data/test/test.set.tgt"]
train = ["./data/train/train.set.src","./data/train/train.set.word","./data/train/train.set.tag","./data/train/train.set.pos","./data/train/train.set.tgt"]
class Example(object):
    def __init__(self,idx,source,word,tag,pos,tgt):
        self.idx = idx
        self.source = source
        self.word = word
        self.tag = tag
        self.pos = pos
        self.tgt = tgt
    
class MyDataset(Dataset):
    def __init__(self,t) -> None:
        input = []
        if t == "dev":
            input = dev.copy()
        elif t == "test":
            input = test.copy()
        else :
            input = train.copy()
        
        if input != None:
            self.src_list = self.fileloader(input[0])
            self.words_list = self.fileloader_strip(input[1])
            self.type_list = self.fileloader_strip(input[2])
            self.start = self.fileloader_strip(input[3])
            self.tgt_list=self.fileloader(input[4])

        #model = os.getcwd()+"/../data/codebert-base"
        self.codeberttokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        #model = os.getcwd()+"/../data/CodeGPT-small-java"
        self.codegptconfig=AutoConfig.from_pretrained('microsoft/CodeGPT-small-java')
        self.codegptconfig.add_cross_attention=True
        self.codegptconfig.is_decoder=True
        self.codegptconfig.bos_token_id=0
        self.codegptconfig.eos_token_id=2

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return outputs
            
        #GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        self.codegpttokenizer = GPT2Tokenizer.from_pretrained('microsoft/CodeGPT-small-java',config=self.codegptconfig)
        token = ["<delete>","<replace>","<self>","<insert>"]
        self.codegpttokenizer.add_tokens(token)
        assert len(self.codegpttokenizer)==50004


    def fileloader(self,filename):
        content = []
        with open(filename,encoding='utf-8') as file:
            content = [line.rstrip().lower() for line in file]
        return content

    def fileloader_strip(self,filename):
        content = []
        with open(filename,encoding='utf-8') as file:
            content = [line.rstrip().lower().split("<<<<>>>>") for line in file]
        return content

    def get_text_embedding(self,line):
        """
        code_tokens=self.codeberttokenizer.tokenize(line)
        tokens=[self.codeberttokenizer.cls_token]+code_tokens+[self.codeberttokenizer.eos_token]
        tokens_ids=self.codeberttokenizer.convert_tokens_to_ids(tokens)
        """
        tokens_ids=self.codeberttokenizer(line,return_tensors="pt")
        return tokens_ids

    def tokenlization(self,line,whole = True):
        if whole:
            return self.codegpttokenizer(line,return_tensors="pt")

        return self.codegpttokenizer(line,is_split_into_words=True,return_tensors="pt")
        

    def __len__(self):
        return len(self.src_list)

    @torch.no_grad()
    def __getitem__(self, item):
        src,words,tags,start = self.src_list[item],self.words_list[item],self.type_list[item],self.start[item]
        source_id = self.get_text_embedding(src)
        w_list = []
        s_list = [0]
        t_list=[]
        for i,w in enumerate(words):
            #prefix space
            token = self.tokenlization(' '+w,True)["input_ids"]
            _,num = token.shape
            for j in range(num):
                t_list.append("<"+tags[i]+">")
                s_list.append(int(start[i]))
        w_list = self.tokenlization(' '.join(["<s>"]+words),True)
        t_list = self.tokenlization(' '.join(["<s>"]+t_list),True)

        assert w_list['input_ids'].size()[1]==t_list['input_ids'].size()[1]==len(s_list)
        return (source_id,w_list,t_list,s_list)
        
    @torch.no_grad()
    def read_examples(self,datasettype):
        examples=[]
        idx = 0
        for line1,line2,line3,line4,line5 in zip(self.src_list,self.words_list,self.type_list,self.start,self.tgt_list):
            examples.append(Example(idx = idx,source=line1,word=line2,tag=line3,pos=line4,tgt=line5))
            idx+=1
        return examples
     
    @torch.no_grad()
    def convert_examples_to_features(self,examples, source_max_length,target_max_length,flag='train'):
        features = []
        if self.codegpttokenizer.pad_token_id is None or self.codeberttokenizer.pad_token_id is None:
            raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.")  
        for example_index, example in enumerate(examples):
            if example_index % 1000==0:
                print(f'convert_examples_to_features {example_index} / {len(examples)}')
            if flag=='train':
                source_ids,w_list,t_list,s_list=self.__getitem__(example.idx)
                source_id=source_ids["input_ids"].tolist()[0]
                source_mask=source_ids["attention_mask"].tolist()[0]
                word_id=w_list['input_ids'].tolist()[0]
                word_mask=w_list['attention_mask'].tolist()[0]
                tag_id=t_list['input_ids'].tolist()[0]
                tag_mask=t_list['attention_mask'].tolist()[0]
                padding_length = target_max_length - len(word_id)
                padding_length_source = source_max_length-len(source_id)
                source_id+=[self.codeberttokenizer.pad_token_id]*padding_length_source
                source_mask+=[0]*padding_length_source
                word_id+=[self.codegpttokenizer.pad_token_id]*padding_length
                word_mask+=[0]*padding_length
                tag_id+=[self.codegpttokenizer.pad_token_id]*padding_length
                tag_mask+=[0]*padding_length
                s_mask=[1]*len(s_list)
                s_list+=[0]*padding_length
                s_mask+=[0]*padding_length
                assert len(source_id)==len(source_mask)==source_max_length
                assert len(word_id)==len(word_mask)==len(tag_id)==len(tag_mask)==target_max_length
                features.append(({'input_ids':source_id,'attention_mask':source_mask},
                    {'input_ids':word_id,'attention_mask':word_mask},
                    {'input_ids':tag_id,'attention_mask':tag_mask},
                    {'input_ids':s_list,'attention_mask':s_mask}))
            else:
               example.word=' '.join(example.word)
               example.tag=' '.join(['<'+i+'>' for i in example.tag])
               features.append((torch.tensor(self.get_text_embedding(example.source)['input_ids']),example.source,example.tgt,example.word,example.tag,example.pos))
        print(f'load dataset complete')
        return features
