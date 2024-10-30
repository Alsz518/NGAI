import os
import re
import yaml
import base64
import logging
import urllib.parse
# from langchain import OpenAI
from langchain.chat_models import AzureChatOpenAI


from LLMAgent.ConversationBot import ConversationBot
from LLMAgent.trafficTools import (
    generatearoadnetworkmap,
    satellite_pic_road_extraction
)

from . import base_session

# os.environ["OPENAI_API_KEY"] = "sk-kcfJcDXKztSEuMxaSqVjvuniMFIlz8HSr2xApuxivkNINiEc"
# os.environ["OPENAI_API_BASE"] = "https://key.langchain.com.cn/v1"
# os.environ["OPENAI_API_PREFIX"] = "https://key.langchain.com.cn"

OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['OPENAI_API_VERSION']
os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['OPENAI_API_BASE']
os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_API_KEY']

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

def generate_data_url(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    ext = os.path.splitext(filename)[-1].lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    content_type = 'image/' + ext
    return 'data:%s;base64,%s' % (
        content_type,
        urllib.parse.quote_from_bytes(base64.b64encode(data), safe='')
    )


class ChatBotSession(base_session.BaseUserSession):
    logger = logging.getLogger(__name__)
    data_path = './'

    def __init__(self):
        # llm = OpenAI(temperature=0)
        super().__init__()
        self.llm = AzureChatOpenAI(
            deployment_name="Test-trans-01",
            temperature=0,
        )
        self.bot = None

    def _prepare_models(self, figfolder='./temp/fig/', filefolder='./temp/file/'):
        self.tool_models = [
            generatearoadnetworkmap(figfolder,filefolder),
            satellite_pic_road_extraction(figfolder,filefolder)
        ]

    def start(self, client_id: str, **kwargs):
        self.logger.debug("start")
        self._prepare_models()
        self.bot = ConversationBot(self.llm, self.tool_models, botPrefix, verbose=True)
        self.logger.debug("bot loaded")
        self.bot.agent_memory.clear()
        self.bot.ch.memory.clear()
        self.started = True
        self.logger.debug("memory cleared, tls reseted")

    def respond(self, message: str):
        #time.sleep(100000)
        sent = {"dir": 1, "type": "text", "msg": message}
        result = [sent]
        self.chat_history_new.append(sent)
        res, cb = self.bot.dialogue(message)
        for memory in self.bot.ch.memory:
            result.append({"dir": 2, "type": "thought", "msg": memory})
        regex = re.compile(r'`([^`]+)`')
        match = regex.search(res)
        if match:
            filename = match.group(1)
            # res = regex.sub('is shown as above', res)
            result.append({"dir": 2, "type": "text", "msg": res})
            result.append({"dir": 2, "type": "img", "msg": generate_data_url(filename)})
        else:
            result.append({"dir": 2, "type": "text", "msg": res})
        self.chat_history_new.extend(result)
        return result

    def reset(self):
        super().reset()
        self.bot.agent_memory.clear()
        self.bot.ch.memory = []

