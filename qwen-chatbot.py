import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys
import os

# --- 系统提示词 ---
# 定义助手的角色和行为准则
system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should be engaging and fun. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# --- 设备和模型加载 ---
# 设置设备（优先使用GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 定义要使用的模型名称
# 可选: "Qwen/Qwen1.5-0.5B-Chat", "Qwen/Qwen1.5-1.8B-Chat", "Qwen/Qwen1.5-7B-Chat"
model_name = "Qwen/Qwen1.5-1.8B-Chat"
print(f"Loading {model_name}...")

# 加载分词器 (Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载模型并手动移动到指定设备
print("Loading model in standard mode and moving to device...")
start_time = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)  # 手动将模型移动到 'cuda' 或 'cpu'
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    sys.exit(1)  # 模型加载失败，退出程序

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")


# --- 核心功能函数 ---
def generate_response(user_input, chat_history=None):
    """根据用户输入和对话历史生成回复"""
    if chat_history is None:
        chat_history = []

    # 1. 格式化对话历史
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})

    # 2. 应用聊天模板并进行分词
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3. 生成模型回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id  # 使用eos_token_id作为pad_token_id
        )

    # 4. 解码并清理回复
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 使用 'assistant' 作为分割点，取最后一部分
    parts = full_response.split('assistant\n')
    if len(parts) > 1:
        # 取最后一个 'assistant\n' 之后的所有内容，并去除首尾的空白
        assistant_response = parts[-1].strip()
    else:
        assistant_response = "Sorry, I couldn't parse the response correctly."

    return assistant_response


def cli_chat():
    """启动命令行聊天循环"""
    print("\n=== Starting CLI Chat (type 'exit', 'quit', or 'q' to end) ===")
    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                # 退出时，打印 Chat History 的所有内容
                print("\n--- Chat History ---")
                if not chat_history:
                    print("No messages in this session.")
                else:
                    for message in chat_history:
                        # 格式化输出，使其更易读
                        role = message["role"].capitalize()
                        content = message["content"]
                        print(f"[{role}]: {content}\n")
                print("--------------------")

                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            print("Assistant: ", end="", flush=True)  # 使用 flush 确保 "Assistant: " 立即显示

            start_time = time.time()
            response = generate_response(user_input, chat_history)
            end_time = time.time()

            print(f"{response}")
            print(f"(Generated in {end_time - start_time:.2f} seconds)")

            # 更新对话历史 维持记忆
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            import traceback
            traceback.print_exc()


def quick_test():
    """执行一个快速测试以确保所有组件正常工作"""
    test_question = "What can you help me with?"
    print(f"\n--- Running Quick Test ---")
    print(f"Test Question: {test_question}")

    start_time = time.time()
    try:
        response = generate_response(test_question)
        end_time = time.time()

        print(f"Response: {response}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        print("--- Test Successful ---")
        return True
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("--- Test Failed ---")
        return False


# --- 主程序入口 ---
def run_assistant():
    """运行助手的主函数"""
    if quick_test():
        # 测试成功后直接启动命令行聊天
        cli_chat()
    else:
        print("\nCould not start the assistant due to test failure.")
        print("Please check the errors above and try again.")


if __name__ == "__main__":
    run_assistant()
