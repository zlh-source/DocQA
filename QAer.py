import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from Rerank import Rerank
import functools
import gradio as gr
import os, json

class QAer:
    def __init__(self, qa_pipeline, reranker, embed_bsz, topk):
        # embedding模型
        self.reranker = Rerank(reranker, qa_pipeline.tokenizer, embed_bsz, topk)
        self.device = torch.device("cuda")
        self.embed_bsz = embed_bsz
        self.qa_instruict = "请参考给定的知识库(文档片段或论文片段)对用户的问题进行回复。json格式的知识库(文档片段或论文片段)信息如下："
        self.knowledge_name = "知识库(文档片段或论文片段)"
        self.system_reply = "好的，请告诉我您有什么问题，我会根据知识库(文档片段或论文片段)为您提供答案。"
        # 大模型
        self.qa_pipeline = qa_pipeline
        

    @torch.no_grad()
    def get_response(self, cur_history , max_tokens = 1024):
        '''
        cur_history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]
        '''
        prompt = self.qa_pipeline.tokenizer.apply_chat_template(cur_history, tokenize=False, add_generation_prompt=True)

        # Generate the summary using the pipeline
        # print(prompt)
        if not self.qa_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") is None: # llama3-chat\
            outputs = self.qa_pipeline(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id = [self.qa_pipeline.tokenizer.eos_token_id, self.qa_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        else:
            outputs = self.qa_pipeline(
                prompt,
                max_new_tokens=max_tokens)
        response = outputs[0]["generated_text"][len(prompt):].strip()
        
        return response
    
    def pdf_submit(self, pdf, msg):
        '''
        提交pdf后执行的动作
        '''
        if pdf is None: # 没有输入pdf, 则重置knowledge
            self.reranker.init_knowledge()
            return None, msg
        
        pdf = pdf[0]
        knowledge_info = self.reranker.pdfReader.get_knowledge(pdf)
        # 将pdf的解析结果进行下载
        if not self.reranker.pdfReader.pdf_path is None and self.reranker.pdfReader.pdf_path!="":
            pdf_name = os.path.split(self.reranker.pdfReader.pdf_path)[-1].replace('.pdf','')
            struct_data_dir = os.path.join(self.reranker.pdfReader.pdf_dir, pdf_name)
            if not os.path.exists(struct_data_dir):
                os.makedirs(struct_data_dir)
            txt_path = os.path.join(struct_data_dir, f"{pdf_name}.txt")
            with open(txt_path, 'w') as fw:
                for line in self.reranker.pdfReader.knowledges_list:
                    print(line, file=fw)
        else:
            txt_path = None
        
        # 做一下摘要
        if knowledge_info=='update knowledge':
            self.reranker.pdfReader.summary = self.summary()
            # gr.Info("接收到文档，摘要如下："+self.reranker.pdfReader.summary)
        return txt_path, msg
        
    def get_knowledge_from_reranker(self, message):
        # 初次选择知识

        selected_knowledge, scores = self.reranker.rerank(message)
        num_selected_knowledge = len(selected_knowledge)
        if num_selected_knowledge == 0:
            selected_knowledge = ['' for _ in range(self.reranker.topk)]
            scores = [None]*self.reranker.topk
        # import pdb;pdb.set_trace()
        if num_selected_knowledge < self.reranker.topk:
            
            selected_knowledge = selected_knowledge + ['' for _ in range(self.reranker.topk - num_selected_knowledge)]
            scores = scores + [None] * (self.reranker.topk - num_selected_knowledge)

        scores = ["%.6lf"%(score) if not score is None else score for score in scores]
        
        return [message] + selected_knowledge + scores
        # return selected_knowledge

    def have_selected_knowledge(self, selected_knowledge):
        # 判断是否存在知识
        for item in selected_knowledge:
            if not item is None and len(item.strip())!=0:
                return True
        return False

    def respond_simplev1(self, query, topk_gr, selected_knowledge, max_tokens=1024):
        llm_input = self.get_llm_input([], selected_knowledge, query, all_context=True) # 用于模型输入
        response = self.get_response(llm_input, max_tokens=max_tokens)
        return response
    

    def respond_simplev2(self, query, max_tokens=1024):
        llm_input = [{"role": "user", "content": query}]
        response = self.get_response(llm_input, max_tokens=max_tokens)
        return response

    def summary(self):
        
        if self.reranker.pdfReader.summary=='' and self.reranker.pdfReader.document!="":
            tokenids = self.qa_pipeline.tokenizer.encode(self.reranker.pdfReader.document)
            docs_summary=""
            for summary_num in [2560, 1280]: # 防止oom
                # print("生成摘要：", summary_num)
                try:
                    docs = self.qa_pipeline.tokenizer.decode(tokenids[:summary_num])
                    docs_summary = self.respond_simplev2(f"请简要说明一些这个文档的主要内容（大约500个字即可）。文档：{docs}", max_tokens=512)
                    break
                except:
                    continue
            return docs_summary
        else:
            return ""


    def respond(self, message, chat_history, *selected_knowledge, respond_again=False):
        
        if not respond_again:
            if message is None or len(message.strip())==0:
                return chat_history, ""
        
        if respond_again: # 撤销上一轮对话
            message = chat_history[-2][0]
            assert not message is None
            chat_history = chat_history[:-2]
        
        topk_gr = 10

        if self.have_selected_knowledge(selected_knowledge):
            selected_knowledge = selected_knowledge[:topk_gr]

        bot_message = self.predict(message, chat_history, selected_knowledge)
        
        chat_history.append((message,None))
        chat_history.append((None, bot_message))
        
        return chat_history, ""

    def merge_strings(self, string_list, min_overlap=32):
        def merge_strings_two(x, y):
            for i in range(len(x)):
                if x[i:] == y[:len(x) - i] and len(x[i:])>min_overlap:
                    return x + y[len(x) - i:]
            return None
        merged_list = [string_list[0]]
        for index in range(1,len(string_list)):
            
            x = merged_list[-1]
            y = string_list[index]
            xy = merge_strings_two(x, y)
            if xy is None:
                merged_list.append(y)
            else:
                merged_list[-1]=xy
        
        return merged_list

    def get_llm_input(self, history, knowledges, user_input, all_context=True):
        '''
            获取大模型的输入
            history: 对话历史
            system_reply: 
            qa_instruict: 以下是给定的知识库，请根据知识库回答用户的问题。注意，你应该尽可能地参考知识库进行回答。
            knowledge_name: 知识库
            knowledges: 检索地结果
            user_input: 用户地当前输入
        '''
        def get_text(text):
            if all_context:
                return text
            if len(text)<15:
                text = text
            else:
                text = text[:5] + "..." + text[-5:]
            return text
        
        # 将前端记录的历史送过来，生成对话历史
        messages_his = []
        for question, answer in history[-6:]: # 仅看最新的3轮对话
            if not question is None:
                messages_his.append({"role": "user", "content": get_text(question)})
            if not answer is None:
                messages_his.append({"role": "assistant", "content": get_text(answer)})
        
        prefix = []
        
        if self.have_selected_knowledge(knowledges): # 存在知识，将知识加入prefix
            
            if False not in [know in self.reranker.pdfReader.knowledge2id for know in knowledges]: # 对知识做排序
                id_know = [(self.reranker.pdfReader.knowledge2id[know], know) for know in knowledges]
                id_know = sorted(id_know, key=lambda x: x[0])
                knowledges = [know for _, know in id_know]
                knowledges = self.merge_strings(knowledges, min_overlap=32) # 合并滑动切分中重合的部分。
                
            prompt = f'{get_text(self.qa_instruict)}\n\n'
            
            knowledge_info = {"知识库(文档或论文)的信息摘要": self.reranker.pdfReader.summary}

            for k_enum, knowledge in enumerate(knowledges): # 这里的知识可能会给的多`
                knowledge_info[f'{self.knowledge_name} {k_enum}'] = get_text(knowledge)
            prompt += str(knowledge_info)
            prefix = prefix + [{"role": "user", "content": prompt}, {"role": "assistant", "content": self.system_reply}] # prefix中加上检索的knowledge

        llm_input = prefix + messages_his + [{"role": "user", "content": user_input}]
        return llm_input

    @torch.no_grad()
    def predict(self, user_input, history, knowledges):
        '''输入问题和对话历史，返回检索的知识+大模型输出'''


        llm_input = self.get_llm_input(history, knowledges, user_input, all_context=True) # 用于模型输入
        # llm_input_log = self.get_llm_input(history, knowledges, user_input, all_context=False) # 用于显示到前端界面
        
        # gr.Info("<SEP>".join([f"{info['role']}: {info['content']}" for info in llm_input_log]))
        
        response = self.get_response(llm_input)
        
        return response
