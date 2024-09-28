import string 
import torch 
import torch.nn as nn


from transformers import BertPreTrainedModel , BertModel, BertTokenizerFast


Device = "cuda" if torch.cuda.is_available() else "cpu"


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric ="cosine"):
        
        super(ColBERT,self).__init__(config)
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric 
        self.dim = dim 
         
        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias =False)

        self.init_weights()


    def forward(self,Q,D):
        return self.score(self.query(*Q),self.doc(*D))
    
    def query(self,input_ids, attention_mask):
        input_ids ,attention_mask = input_ids.to(Device),attention_mask.to(Device)
        Q = self.bert(input_ids,attention_mask = attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q ,p=2,dim=2)
    
    def doc(self,input_ids, attention_mask ,keep_dims = True):
        input_ids,attention_mask = input_ids.to(Device),attention_mask.to(Device)
        D = self.bert(input_ids,attention_mask= attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=Device).unsqueeze(2).float()

        D = D * mask 

        D = torch.nn.functional.noramlize(D,p=2, dim =2)

        return D 
    

    def score(self, Q,D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
        

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask





       

    


    



