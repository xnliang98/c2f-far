import os
import torch
import numpy as np
import yaml
from tqdm.auto import tqdm
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, TensorDataset, Dataset

from transformers import AutoConfig, AutoTokenizer, AutoModel

from utils import load_jsonline, save_jsonline, load_txt, save_txt, make_dirs


def flatten_documents(documents):
    doc_indices = [] # (start sentence index, end sentence index) \in document
    sentences = []
    for document in documents:
        doc_indices.append((len(sentences), len(sentences) + len(document)))
        sentences.extend(document)
    assert len(sentences) == doc_indices[-1][1]
    return sentences, doc_indices

class TextLoader(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = sentences
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        outputs = self.tokenizer(self.data[idx], 
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True, 
                        add_special_tokens=True, 
                        return_tensors='pt')
        all_inputs = {}
        all_inputs["input_ids"] = outputs["input_ids"].squeeze()
        all_inputs["attention_mask"] = outputs["attention_mask"].squeeze()
        return all_inputs

def pooling(model_output, attention_mask, pooling_type="mean"):
    token_embeddings = model_output
    
    input_mask_expanded = attention_mask.eq(0).unsqueeze(2)
    if pooling_type == "mean":
        if attention_mask is not None:
            token_embeddings = token_embeddings.masked_fill(input_mask_expanded, 0)
        sum_embeddings = torch.sum(token_embeddings, dim=1)
        pooled_output = sum_embeddings / (input_mask_expanded.size(1) - input_mask_expanded.float().sum(1))
    else:
        if attention_mask is not None: 
             token_embeddings = token_embeddings.masked_fill(input_mask_expanded, 1e-9)
        pooled_output = torch.max(token_embeddings, dim=1)[0]
        
    return pooled_output

def get_sentence_embedding(model, tokenizer, sentences, batch_size=1024, max_length=60, middle_layers=[], method="cls", device='cpu'):
    text_loader = TextLoader(sentences, tokenizer, max_length)
    dataloader = DataLoader(text_loader, batch_size=batch_size, shuffle=False, num_workers=24)

    embeddings = []
    for batch in tqdm(dataloader, desc='encoding'):
        for k, v in batch.items():
            batch[k] = v.to(device) 

        with torch.no_grad():
            hidden_states, cls_outputs, all_hidden_states = model(**batch, output_hidden_states=True, return_dict=False)
            # print(hidden_states.shape, cls_outputs.shape)
#             hidden_states, cls_outputs, all_hidden_states = model(input_ids=inputs['input_ids'], attention_mask=batch[1], output_hidden_states=True, return_dict=False)
            all_hidden_states = all_hidden_states[0]
#             print(all_hidden_states.shape, type(all_hidden_states))
            if middle_layers != []:
                if len(middle_layers) == 1:
                    hidden_states = all_hidden_states[middle_layers[0]]
                else:
                    hidden_states = torch.mean(all_hidden_states[middle_layers], dim=0)
            if method == 'cls':
                embeddings.append(cls_outputs.detach().cpu().numpy())
            elif method == 'max':
                embeddings.append(pooling(hidden_states, batch["attention_mask"], 'max').detach().cpu().numpy())
            elif method == 'mean':
                embeddings.append(pooling(hidden_states, batch["attention_mask"], 'mean').detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    print(embeddings.shape)

    assert len(embeddings) == len(sentences)
    return embeddings

def reconstruct_document(doc_indices, sentences_embeddings):
    documents = []
    for start, end in doc_indices:
        documents.append(sentences_embeddings[start: end])
        # print(sentences_embeddings[start: end])
    
    assert len(doc_indices) == len(documents)
    return documents

def main(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    dataset = args['dataset']
    data_type = args['data_type']
    model_name_or_path = args['model_name_or_path']

    data_path = args['data_path']
    output_path = args['output_path']

    method = args['method']
    batch_size = args['batch_size']
    max_length = args['max_length']
    middle_layers = args['middle_layers']

    print("Config:\n", args)
    print("Load data and model.")
    data = load_jsonline(os.path.join(data_path, dataset, f"{data_type}.json"))
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs are used.")
        model = DataParallel(model)
    model.to(device)
    
    documents = [item['document'].strip().split("\t") for item in data]
    sentences, doc_indices = flatten_documents(documents)
    
    print("Begin to encode the whole document...")
    sentences_embeddings = get_sentence_embedding(model, tokenizer, sentences, batch_size, max_length, middle_layers, method, device)
    documents = reconstruct_document(doc_indices, sentences_embeddings)
    cache_path = os.path.join(output_path, dataset, "sentence_embeddings")
    special_name = f"{data_type}.{method}.{model_name_or_path.split('/')[-1]}.{max_length}.pt"
    make_dirs(cache_path)
    print("Saving ...")
    torch.save(documents, os.path.join(cache_path, special_name))
    print("Encoding Finished.")

if __name__ == "__main__":
    import sys
    yaml_path = sys.argv[1]
    main(yaml_path)