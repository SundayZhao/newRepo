import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import modeling_outputs 

class PointerNetwork(nn.Module):
  def __init__(self, target_length,dim):
    super(PointerNetwork, self).__init__()
    self.encoder_dense = nn.Linear(dim, dim)
    self.decoder_dense = nn.Linear(dim, dim)
    self.ln_y=nn.LayerNorm([target_length],eps=0.000001,elementwise_affine=True)
    self.criterion = torch.nn.CrossEntropyLoss()
    self.drop_out=nn.functional.dropout
    self.t_cache=[]
    self.tagmodel_cache=None
    self.config = AutoConfig.from_pretrained("microsoft/codebert-base")
    self.step=0
    
  def gather_nd(self,params, indices):
    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    for i in range(ndim)[::-1]:
      idx += indices[i] * m
      m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

  def prepare_inputs_for_generation( self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
    model_inputs = {"input_ids": input_ids,'past_key_values':None}
    return model_inputs
        
  def forward(self, encoder_output=None, decoder_output=None,labels=None,**kwargs):
    if encoder_output==None or decoder_output==None:
        #decodermodel
        #encoder_embedding,t_encode
        ans = []
        t_encode=self.t_cache[0]
        t_list = torch.tensor([t_encode[:self.step+1]]).to(kwargs['encoder_hidden_states'].device)
        decode_result = self.tagmodel_cache(input_ids=t_list,attention_mask=t_list.ne(0),encoder_hidden_states=kwargs['encoder_hidden_states'],output_hidden_states=True)
        decode_hidden = decode_result["hidden_states"][-1]
        q=self.decoder_dense(decode_hidden)
        k=self.encoder_dense(kwargs['encoder_hidden_states'])
        scalar = torch.rsqrt(torch.tensor(q.size()[2],dtype=torch.float))
        logits = torch.matmul(q * scalar, torch.transpose(k,1,2))
        self.step=self.step+1
        return modeling_outputs.CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=kwargs['past_key_values'],
            hidden_states=kwargs['encoder_hidden_states'],
            attentions=None,
            cross_attentions=None,
        )
    q=self.decoder_dense(decoder_output)
    k=self.encoder_dense(encoder_output)
    scalar = torch.rsqrt(torch.tensor(q.size()[2],dtype=torch.float))
    logits = torch.matmul(q * scalar, torch.transpose(k,1,2))
    #gathern
    batch_size = encoder_output.size()[0]
    num_indices =labels.size()[1]
    batch_indices =torch.tile(torch.arange(batch_size).unsqueeze(1),[1,num_indices])
    batch_indices=batch_indices.cuda()
    gather_nd_indices = torch.stack([batch_indices, labels], 2)
    pointer_out=self.gather_nd(encoder_output,gather_nd_indices )
    #loss
    xent=self.criterion(torch.transpose(logits,1,2),labels)
    loss=xent
    #loss=torch.sum(xent*loss_mask)
    #x,y
    y=self.ln_y(pointer_out)
    y=self.drop_out(y,p=0.2)
    b=y.unsqueeze(-1) 
    decoder_output= y.unsqueeze(-1)+decoder_output
    return decoder_output,logits,loss