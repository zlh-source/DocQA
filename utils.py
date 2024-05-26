from transformers import pipeline
import torch


def move_letter(move, text_up, text_down):
    # 检索顺序做前后调换
    if move=='Up':
        text_down, text_up = text_up, text_down
    elif move=='Down':
        text_up, text_down = text_down, text_up

    return text_up, text_down


def jump_fn(*jump_text):
    # 修改检索顺序
    topk = len(jump_text)//2
    jump = list(jump_text[:topk])
    texts = list(jump_text[topk:])
    # 先校验jump的格式：仅有有一个有数字，并且为[0,topk-1]
    from_index=None
    target_index=None
    try:
        for index, item in enumerate(jump):
            if len(item)!=0:
                if from_index is None and target_index is None:
                    from_index = index
                    target_index = int(item)
                    assert target_index>=0 and target_index<topk
                else:
                    assert False
    except:
        return ['']*topk + texts
    
    element = texts.pop(from_index)  # 移除第index个元素，并获取它
    texts.insert(target_index, element)  # 将这个元素插入到target位置
    return ['']*topk + texts

def get_pipeline(model_path):
    qa_pipeline = pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use "cpu" if CUDA is not available
    )
    if qa_pipeline.tokenizer.pad_token is None:
        qa_pipeline.tokenizer.pad_token = qa_pipeline.tokenizer.eos_token
    return qa_pipeline
