import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator, List

random.seed(42)

def read_texts_from_jsonl_batched(
    file_path: str,
    batch_size: int = 2000,
    max_chars_per_text: int = 4096,
    max_samples: int = 500000,
    sample_ratio: float = None,
) -> Generator[List[str], None, None]:
    batch = []
    seen = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if seen >= max_samples:
                break

            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text:
                    continue

                if sample_ratio is not None and random.random() > sample_ratio:
                    continue

                text = text[:max_chars_per_text]
                batch.append(text)
                seen += 1

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except Exception:
                continue

    if batch:
        yield batch

def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """训练并保存自定义tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # 添加文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 配置特殊token
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
    vocab_size=6144,
    special_tokens=["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
    min_frequency=10,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 训练tokenizer
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl_batched(
    data_path,
    batch_size=2000,
    max_chars_per_text=4096,
    max_samples=500000
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 创建配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)

def main():
    # 配置路径
    data_path = "/fs-computility-new/UPDZ11_zhanglujia/zhangchuanxi.p/llm-demo/LLamA2_demo/data/mobvoi_seq_monkey_general_open_corpus.jsonl"
    save_dir = "/fs-computility-new/UPDZ11_zhanglujia/zhangchuanxi.p/llm-demo/LLamA2_demo/tokenizer"

    # 训练tokenizer
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144
    )

    # 评估tokenizer
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()