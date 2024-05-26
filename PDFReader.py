from tqdm import tqdm
import fitz  # PyMuPDF
import re
import hashlib
import shutil
import zipfile
import json
import datetime
import random
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader

class PDFReader:
    def __init__(self, lm_tokenizer):
        self.pdf_dir = './pdfs/'
        self.lm_tokenizer = lm_tokenizer
        self.init_knowledge()
        
    def init_knowledge(self):
        self.knowledges_list = []
        self.id2knowledge = {}
        self.knowledge2id = {}
        self.pdf_path = ""
        self.summary = ""
        self.document = ""

    def have_knowledge(self):
        return len(self.knowledges_list)!=0 and self.pdf_path not in ["", None]
        
    def get_md5(self, pdf_path):
        return hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

    def sliding_split(self, document):
        document = " ".join(document.split())
        document_token_ids = self.lm_tokenizer.encode(document)
        sliding_split_res = []
        for left in range(0, len(document_token_ids), 128):
            cur_token_ids = document_token_ids[left: left+256]
            sliding_split_res.append(self.lm_tokenizer.decode(cur_token_ids))
        return sliding_split_res

    def get_pdfer(self, pdf_path):
        # 读取pdf
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        document = " ".join([page.page_content for page in pages])
        document = document.replace('\n', '').replace('\t', '')
        all_knowledge = self.sliding_split(document)
        
        # result = self.splitter(documents = document)
        
        # all_knowledge = []
        # for text in result['text'].split("\n\t"):
        #     if text not in all_knowledge:
        #         if len(text)<=5: # 太短的不要
        #             continue
        #         if len(text)<512:
        #             all_knowledge.append(text)
        #         else: # 太长的滑动切分
        #             all_knowledge.extend(self.sliding_split(text))
        return all_knowledge, document
    

    def pdf_to_txt(self, pdf):
        '''将pdf转为txt,对于部分pdf会解析失败'''
        
        # 打开PDF文件
        pdf = fitz.open(pdf)
        # 创建一个文本变量来存储所有页面的内容
        knowledges = []
        # 遍历PDF的每一页
        for page_num in range(len(pdf)):
            # 获取页面
            page = pdf[page_num]
            # 提取页面的文本
            try:
                knowledges.append(page.getText().strip())
            except:
                knowledges.append(page.get_text().strip())
        # 关闭PDF文件
        pdf.close()
        knowledges=" ".join(knowledges).replace('\n', " ")        
        return self.sliding_split(knowledges), knowledges
        
    def pdf_to_knowledge(self, pdf):
        
        # 获取当前时间并格式化为字符串
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # 设置保存路径，这里假设在当前目录
        save_path = f"{self.pdf_dir}/{current_time}.pdf"
        shutil.copy(pdf.name, save_path)
        
        if len(self.knowledges_list)==0: # 不存在all_paragraph 还要看pdf是否更换了
            try:
                all_paragraph, document = self.get_pdfer(save_path) # 对于部分pdf 可能会解析失败
            except:
                all_paragraph, document = self.pdf_to_txt(save_path)
        else:
            all_paragraph = self.knowledges_list
            document = self.document

        return save_path, all_paragraph, document
        
    def get_knowledge(self, pdf):
        if pdf is None: #没有上传pdf
            self.init_knowledge()
            return "no knowledge"
        
        # 判断是否已经存在相同的知识
        if self.pdf_path != "" and self.get_md5(self.pdf_path)==self.get_md5(pdf):
            # print("已经存在完全相同知识")
            # gr.Info("使用已上传的知识库")
            return "have same knowledge"
        else: # 所有知识全部清空
            self.init_knowledge()

        self.pdf_path, self.knowledges_list, self.document = self.pdf_to_knowledge(pdf)
        
        for index, knowledge in enumerate(self.knowledges_list):
            self.id2knowledge[str(index)] = knowledge
            self.knowledge2id[knowledge] = index
        return "update knowledge"
