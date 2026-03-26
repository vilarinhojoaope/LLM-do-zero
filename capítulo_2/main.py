import importlib
import importlib.metadata
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

context_size = 6

with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]

for i in range(1,context_size+1):
    context = enc_sample[:i]
    target = enc_sample[i]

    print(tokenizer.decode(context)," ---->",tokenizer.decode([target]))

class GPTDatasetV1(Dataset):
    
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []

        #token id sao os tokens da entrada de texto
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
        assert len(token_ids)>max_length, "O numero de entradas tokenizadas deve ser maior ou igual a max_length+1 "

        #percorre toda a entrada de texto e vai criando uma lista
        #de tensors para input enquanto for possivel criar conjuntos
        #de tensores do tamanho max_length

        #fazemos o for iterar até len(token_ids) - max_length
        #pois o i é o ponto inicial de captura de um chunk de 
        #max_length de tamanho

        for i in range(0,len(token_ids) - max_length,stride):
            #captura um conjunto de tokens a partir do token i atual
            #ex: i=3,max_length=2, logo capturamos os tokens de 3 ate 5
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1: i+max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
    
dataloader = create_dataloader_v1(raw_text,batch_size=2,max_length=4,stride=1,shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

#Em resumo:
# no dataset criamos a base de dados em lista de tensores de
# input ids e target ids, ou seja, no dataset temos todos os
# tensores do texto, no create_dataloader_v1 é realizada a
# criação do dataloader, ou sjea, o carregador de dados que
# captura do dataset uma configuração específica de amostragem,
# exemplo, batch_size = 2, então o proprio objeto dataloader
# captura do dataset 2 batchs de input e target cada um

#Creating token embeddings:
torch.manual_seed(42)
#vocab_size é a quantidade de palavras do vocabulario
#output_dim é a dimensão do embedding da palavra (token de entrada)
vocab_size  = 50257
output_dim = 256

data_iter = iter(dataloader)
inputs,targets = next(data_iter)

#cria uma matriz de embedding onde cada linha é um token do vocabulario
#e cada token é representado por um vetor de 256 posições (embeddings)
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
token_embeddings = token_embedding_layer(inputs)
