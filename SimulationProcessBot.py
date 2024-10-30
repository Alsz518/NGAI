import os
import re
import yaml
from rich import print
# from langchain import OpenAI
from langchain.chat_models import AzureChatOpenAI

from LLMAgent.ConversationBot import ConversationBot

from LLMAgent.trafficTools import (
    generatearoadnetworkmap
)

import gradio as gr
import openai.api_requestor
openai.api_requestor.TIMEOUT_SECS = 30

# ------------------------------------------------------------------------------
# --ZH 初始化 LLM
# --EN Initialize a LLM
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['OPENAI_API_VERSION']
os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['OPENAI_API_BASE']
os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_API_KEY']

llm = AzureChatOpenAI(
    deployment_name="Test-trans-01",
    temperature=0,
)

# ------------------------------------------------------------------------------
# --ZH 初始化工具
# --EN Initialize the tools
toolModels = [
    generatearoadnetworkmap()
]

# ------------------------------------------------------------------------------
# --ZH 定义 prompts，催眠 LLM，让 LLM 了解工作内容，减少幻觉
# --EN Define prompts, hypnotize LLM, let LLM understand the work content
#      and reduce hallucinations

botPrefix = """
[WHO ARE YOU]
You are an AI assistant that aids humans in traffic simulation modeling. Your job is to tune the appropriate tools to complete the simulation modeling task according to the modeling needs of the human user.
[YOUR ACTION GUIDLINES]
1. The action to take should be one of the tool names, but not a sentence. For example: Action: Get Intersection Performance
2. If you can not find any appropriate tool for your task, try to do it using your own ability and knowledge as a chat AI.
[THINGS YOU CANNOT DO]
You are forbidden to fabricate any tool names.
You are forbidden to fabricate any input parameters when calling tools!
Your Final Answer can never be N/A.
"""

# ------------------------------------------------------------------------------
# --ZH 初始化对话模型
# --EN Initilize the ConversationBot
bot = ConversationBot(llm, toolModels, botPrefix, verbose=True)

# ------------------------------------------------------------------------------
# --ZH 设置 gradio 界面
# --EN Configure the grdio interface


def reset(chat_history: list, thoughts: str):
    chat_history = []
    thoughts = ""
    bot.agent_memory.clear()
    bot.ch.memory = []
    return chat_history, thoughts


def respond(msg: str, chat_history: list, thoughts: str):
    res, cb = bot.dialogue(msg)
    regex = re.compile(r'`([^`]+)`')
    try:
        filenames = regex.findall(res)
    except AttributeError:
        filenames = None
    if filenames:
        chat_history += [(msg, None)]
        for fn in filenames:
            chat_history += [(None, (fn,))]
        chat_history += [(None, res)]
    else:
        chat_history += [(msg, res)]

    thoughts += f"\n>>> {msg}\n"
    for actionMemory in bot.ch.memory[-2]:
        thoughts += actionMemory
        thoughts += '\n'
    thoughts += f"<<< {res}\n"
    return "", chat_history, thoughts


with gr.Blocks(
    title="Traffic Simulation Process Bot", theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg)
) as demo:
    with gr.Row(visible=True, variant="panel"):
        with gr.Column(visible=True, variant='default'):
            chatbot = gr.Chatbot(scale=2, height=650)

            with gr.Row():
                humanMsg = gr.Textbox(scale=2)
                submitBtn = gr.Button("Submit", scale=1)
            clearBtn = gr.ClearButton()
            gr.Examples(
                label='You may want to ask the following questions:',
                examples=[
                    "帮我生成一个天安门附近1000m的路网图",
                ],
                inputs=[humanMsg],
                # outputs=[humanMsg, chatbot],
                # fn=testFunc
            )
        ReActMsg = gr.Text(
            label="Thoughts and Actions of the Chatbot",
            interactive=False,
            lines=50
        )

    humanMsg.submit(
        respond,
        [humanMsg, chatbot, ReActMsg],
        [humanMsg, chatbot, ReActMsg]
    )
    submitBtn.click(
        respond,
        [humanMsg, chatbot, ReActMsg],
        [humanMsg, chatbot, ReActMsg]
    )
    clearBtn.click(reset, [chatbot, ReActMsg], [chatbot, ReActMsg])

if __name__ == "__main__":
    demo.launch(share=False,)
