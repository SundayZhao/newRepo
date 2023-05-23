from transformers import EncoderDecoderModel, AutoTokenizer,Trainer,GPT2Tokenizer,Seq2SeqTrainer
from transformers import TrainingArguments
import numpy as np
import evaluate
import os
from torch.utils.data import DataLoader
import numpy as np
import torch
from datasets import load_dataset, Dataset,DatasetDict
from datasets import load_from_disk
from datasets.features import Value
from transformers import DataCollatorForSeq2Seq
import evaluate
from transformers import EarlyStoppingCallback

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#配置
#batch_size
train_bs=20
eval_bs=20 
#load_best_model_at_end =True,dataloader_drop_last=True,eval_steps=20,
training_args = TrainingArguments(do_train=True,do_eval=False,learning_rate=2e-5,warmup_steps=5000,per_device_train_batch_size=train_bs,per_device_eval_batch_size=eval_bs,num_train_epochs=20,save_strategy="steps",save_steps =5000,evaluation_strategy="no",save_total_limit =3,output_dir="test_trainer",fp16=True)
#训练集文件位置
train_buggy_file='train.buggy-fixed.buggy'
train_fixed_file='train.buggy-fixed.fixed'
#验证集文件位置
valid_buggy_file='valid.buggy-fixed.buggy'
valid_fixed_file='valid.buggy-fixed.fixed'
#初始化权重，如果为True则加载自己的权重，相当于继续训练，为False则从官方的权重重新训练
from_custome=False
#读取和保存的权重文件夹路径
codebert_weight_path='codebert'
codegpt_weight_path='codegpt'
#是否进行mask训练
do_mask=False
#序列的最大长度，超出则阶段
sequence_maxlen=300
#缓存数据集
dataset_cache=True
es=EarlyStoppingCallback(early_stopping_patience=4)
#断点续训
train_from_ck=False



def compute_metrics(eval_preds):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = np.argmax(logits, dim=-1)
    return pred_ids, labels

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

#分词器
print('正在加载分词器')
tokenizer_codebert = AutoTokenizer.from_pretrained("microsoft/codebert-base")
GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
tokenizer_codegpt = GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-java")
token = ["<delete>","<replace>","<self>","<insert>"]
tokenizer_codegpt.add_tokens(token)
assert len(tokenizer_codegpt)==50004
tokenizer_codebert.save_pretrained(codebert_weight_path)
tokenizer_codegpt.save_pretrained(codegpt_weight_path)




#定义数据集预处理
def preprocess_function(examples):
  inputs = [ex for ex in examples["buggy"]]
  targets = [ex for ex in examples["fixed"]]
  model_inputs = tokenizer_codebert(inputs, max_length=sequence_maxlen, truncation=True)
  labels = tokenizer_codegpt(targets, max_length=sequence_maxlen, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs
  
dataset_prepare=0
if dataset_cache:
  print('正在预处理数据集[缓存]')
  dataset_prepare=load_from_disk("./datasetcache")
  print(dataset_prepare)
else:
  #加载训练集
  print('正在加载训练集')
  buggycode_train_input = open(train_buggy_file, 'r', encoding='utf-8').read().strip().lower().split('\n')
  fixedcode_train_input = open(train_fixed_file, 'r', encoding='utf-8').read().strip().lower().split('\n')
  code_train_list=[]
  for index in range(len(buggycode_train_input)):
    code_train_list.append({'buggy':buggycode_train_input[index],'fixed':fixedcode_train_input[index]})
  dataset_train=Dataset.from_list(code_train_list)


  #加载验证集
  print('正在加载验证集')
  buggycode_valid_input = open(valid_buggy_file, 'r', encoding='utf-8').read().strip().lower().split('\n')[:4000]
  fixedcode_valid_input = open(valid_fixed_file, 'r', encoding='utf-8').read().strip().lower().split('\n')[:4000]
  code_valid_list=[]
  for index in range(len(buggycode_valid_input)):
    code_valid_list.append({'buggy':buggycode_valid_input[index],'fixed':fixedcode_valid_input[index]})

  #构造验证集，训练集
  print('正在预处理数据集')
  dataset_valid=Dataset.from_list(code_valid_list)
  dataset_all=DatasetDict({'train':dataset_train,'validation':dataset_valid})
  dataset_prepare=dataset_all.map(preprocess_function,batched=True,remove_columns=dataset_all["train"].column_names)
  dataset_prepare.save_to_disk("./datasetcache")


print('正在加载模型')
#模型
model=0
if from_custome and len(codebert_weight_path)>0 and  len(codegpt_weight_path)>0:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(codebert_weight_path, codegpt_weight_path)
else:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("microsoft/codebert-base", "microsoft/CodeGPT-small-java")
model.config.decoder_start_token_id = tokenizer_codebert.bos_token_id
model.config.eos_token_id = tokenizer_codebert.eos_token_id
model.config.pad_token_id = tokenizer_codegpt.pad_token_id
model.config.bos_token_id = 0
model.decoder.config.use_cache = False
model.config.max_length = sequence_maxlen
model.config.min_length = 50
model.config.no_repeat_ngram_size = 3
model.length_penalty = 2.0
model.num_beams = 4
if from_custome==False:
    model.decoder.resize_token_embeddings(len(tokenizer_codegpt))
#print('length of codegpt tokenizer:',len(tokenizer_codegpt))

#构造 data_coll
print('正在构造收集器')
data_collator = DataCollatorForSeq2Seq(tokenizer_codebert, model=model,padding='max_length',max_length=sequence_maxlen,label_pad_token_id=1)
#train_dataloader = DataLoader(dataset_prepare["train"],shuffle=True,collate_fn=data_collator,batch_size=train_bs)
#eval_dataloader = DataLoader(dataset_prepare["validation"], collate_fn=data_collator, batch_size=eval_bs)
"""
for i in range(3999):
  batch = data_collator([dataset_prepare["validation"][i]])
  if batch["labels"].size()[1]!=300:
    print(batch["labels"].size())
exit()
"""
#评估器
metric = evaluate.load("accuracy")
#metric = evaluate.load("sacrebleu")
#数据集
#small_train_dataset = MyDataset(train_buggy_file,train_fixed_file,tokenizer_codebert,tokenizer_codegpt,mask_pretrain=do_mask,max_len=500)
#small_eval_dataset = MyDataset(valid_fixed_file,valid_fixed_file,tokenizer_codebert,tokenizer_codegpt,mask_pretrain=do_mask,max_len=500)
#训练器
#callbacks=[es],
#compute_metrics=metric,
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_prepare["train"],
    eval_dataset=dataset_prepare["validation"],
    data_collator=data_collator,
)
print("开始训练")
trainer.train(resume_from_checkpoint=train_from_ck)
#训练完成后单独保存codebert和codegpt的权重,整个模型的权重在test_trainer文件夹内
print("正在导出模型检查点")
#保存训练结果到文件夹，codebert和codegpt分开保存
model.encoder.save_pretrained(codebert_weight_path)
model.decoder.save_pretrained(codegpt_weight_path)
print("finish")