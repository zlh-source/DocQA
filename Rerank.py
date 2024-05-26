import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from PDFReader import PDFReader
import gradio as gr

class Rerank:
    def __init__(self, reranker, lm_tokenizer, embed_bsz, topk):
        # embedding模型
        self.reranker = reranker
        
        self.embed_bsz = embed_bsz
        self.topk = topk
        
        # 读取pdf的对象
        self.pdfReader = PDFReader(lm_tokenizer)
        
    def init_knowledge(self):
        self.pdfReader.init_knowledge()
        
    @torch.no_grad()
    def rerank(self, query):
        '''
        做检索，返回文本和对应的score
        '''
        if not self.pdfReader.have_knowledge() or self.topk==0:
            return [], []
        
        select_knowledges, select_scores = [], []
        all_pairs = [[query, knowledge] for knowledge in self.pdfReader.knowledges_list]
        
        all_scores = []
        
        for index in range(0, len(all_pairs), self.embed_bsz):
            pairs = all_pairs[index: index + self.embed_bsz]
            scores = self.reranker.compute_score(pairs, normalize=True)
            if len(pairs)>1:
                all_scores.extend(scores)
            else:
                all_scores.append(scores)
        scores = torch.tensor(all_scores)
        
        for k_index in scores.topk(min(scores.shape[0], self.topk))[1].tolist():
            select_knowledges.append(self.pdfReader.knowledges_list[k_index])
            select_scores.append(scores[k_index].item())
        return select_knowledges, select_scores