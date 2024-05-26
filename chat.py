import os
import gradio as gr
from QAer import QAer
import functools
import argparse
from FlagEmbedding import FlagReranker
from utils import move_letter, jump_fn, get_pipeline


argparse = argparse.ArgumentParser()
argparse.add_argument('--llm_path', type=str, default='./Qwen1.5-7B-Chat/')
argparse.add_argument('--rerank_path', type=str, default='./bge-reranker-v2-m3/')

args, unknown = argparse.parse_known_args()

model_path = args.llm_path
rerank_path = args.rerank_path

if gr.NO_RELOAD:
    qa_pipeline = get_pipeline(model_path)
    reranker = FlagReranker(rerank_path, use_fp16=True)
    
embed_bsz=16
topk=50
llmer = QAer(qa_pipeline, reranker, embed_bsz, topk)

css = """
#pdfin {background-color: #F2FAFC}
#pdfout {background-color: #BEE6F0}
#chatbot1 {background-color: #ABD1BC}
#chatbot {background-color: #BED0F9}
#button {height: 55px}
#select_info {background-color: #BED0F9}
#msg {background-color: #B3C4D4}
footer {visibility: hidden}
"""

# up {height: 20px; width: 20px; font-size:10px}
# down {height: 20px; width: 20px; font-size:10px}
with gr.Blocks(title='知识库对话系统', css=css) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                pdf = gr.Files(file_types=['.pdf'], label='pdf形式的知识库（删除后则为自由问答）', show_label=True, elem_id="pdfin") # pdf上传
                txt_extract = gr.File(file_types=['.text'], label='pdf解析结果', show_label=True, elem_id="pdfout") # pdf上传

            # topk_gr = gr.Slider(minimum=0, maximum=10, step=1, value=1, label="检索的topk")
            
            # 检索出来的信息
            all_score = []
            all_move = []
            selected_knowledge = []
            all_jump = []
            with gr.Group():
                for topk_index_ in range(0, topk, 5):
                    top_info=f"相关信息：top {topk_index_}-{topk_index_+4}"
                    if topk_index_<10:
                        top_info+=' (作为知识库参与生成)'
                    with gr.Accordion(top_info, open=False):
                        for topk_index in range(topk_index_ , topk_index_ + 5):
                            with gr.Group():
                                text=gr.TextArea(placeholder=f'none', label=f'top-{topk_index}', show_label=True, show_copy_button=True, lines=3)
                                with gr.Row():
                                    score = gr.Textbox(placeholder=f'Score', show_label=False, elem_id="button")
                                    if topk_index==0:
                                        up = gr.Button(f"None", elem_id="button")
                                        down = gr.Button(f"Down", elem_id="button")
                                    elif topk_index == topk-1:
                                        up = gr.Button(f"Up", elem_id="button")
                                        down = gr.Button(f"None", elem_id="button")
                                    else:
                                        up = gr.Button(f"Up", elem_id="button")
                                        down = gr.Button(f"Down", elem_id="button")
                                    jump = gr.Textbox(placeholder=f'Go to 0-{topk-1}', show_label=False, elem_id="button")
                            all_score.append(score)
                            all_move.append([up, down])
                            selected_knowledge.append(text)
                            all_jump.append(jump)
                            
        with gr.Column():
            
            chatbot = gr.Chatbot(label='对话窗口', show_label=True,bubble_full_width=False, \
                height=700, likeable=True, elem_id="chatbot", elem_classes="feedback",\
                    avatar_images=['./imgs/user.png','./imgs/chatbot.png']) # 聊天框
            msg = gr.Textbox(label='✏️ 用户输入', show_label=True, placeholder="", elem_id='msg') # 输入框
            
            with gr.Row():
                # submit = gr.Button(value="Step 1+2. 知识检索 + 生成回复", variant="primary")
                submit_again = gr.Button(value="🛠️ 重新生成")
                reset_button = gr.ClearButton([msg, chatbot, pdf, txt_extract, *selected_knowledge, *all_score], value="🧹 清空对话")
            # stop = gr.Button(value="停止", variant="stop")
            
        
    # 为所有的按鍵添加事件
    for line_index in range(topk):
        if line_index==0:
            all_move[line_index][1].click(fn=move_letter, inputs=[all_move[line_index][1], selected_knowledge[line_index], selected_knowledge[line_index+1]], outputs=[selected_knowledge[line_index], selected_knowledge[line_index+1]])
        elif line_index==topk-1:
            all_move[line_index][0].click(fn=move_letter, inputs=[all_move[line_index][0], selected_knowledge[line_index-1], selected_knowledge[line_index]], outputs=[selected_knowledge[line_index-1], selected_knowledge[line_index]])
        else:
            # up
            all_move[line_index][0].click(fn=move_letter, inputs=[all_move[line_index][0], selected_knowledge[line_index-1], selected_knowledge[line_index]], outputs=[selected_knowledge[line_index-1], selected_knowledge[line_index]])
            # down
            all_move[line_index][1].click(fn=move_letter, inputs=[all_move[line_index][1], selected_knowledge[line_index], selected_knowledge[line_index+1]], outputs=[selected_knowledge[line_index], selected_knowledge[line_index+1]])
    
    for line_index in range(topk):
        all_jump[line_index].submit(fn= jump_fn, inputs = all_jump + selected_knowledge,outputs = all_jump + selected_knowledge)
    
    pdf.change(fn=llmer.pdf_submit, inputs=[pdf, msg], outputs=[txt_extract, msg],queue=True)
    
    # 检索+生成
    msg_click_event = msg.submit(fn = llmer.pdf_submit, inputs = [pdf,msg], outputs = [txt_extract, msg], queue=True).then(\
                fn = llmer.get_knowledge_from_reranker, \
                inputs=[msg], \
                outputs=[msg] + selected_knowledge + all_score,queue=True).then(\
                        fn = functools.partial(llmer.respond, respond_again=False), \
                        inputs=[msg, chatbot, *selected_knowledge], \
                        outputs=[chatbot, msg],queue=True)
    
    # 回滚
    submit_again_click_event = submit_again.click(fn = functools.partial(llmer.respond, respond_again=True), \
                        inputs=[msg, chatbot, *selected_knowledge], \
                        outputs=[chatbot, msg],queue=True)
                        
if __name__ == "__main__":
    # demo.queue().launch(server_port=8000, favicon_path='./imgs/logo.png', show_error=True)
    demo.queue().launch(favicon_path='./imgs/logo.png', show_error=True)