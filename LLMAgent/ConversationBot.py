from rich import print
from typing import Any, List
from langchain import LLMChain
from langchain.agents import Tool
from langchain.chat_models import AzureChatOpenAI
from LLMAgent.callbackHandler import CustomHandler
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor




suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


class ConversationBot:
    def __init__(
            self, llm: AzureChatOpenAI, toolModels: List,
            customedPrefix: str, verbose: bool = False
    ) -> Any:
        self.ch = CustomHandler()
        tools = []

        for ins in toolModels:
            func = getattr(ins, 'inference')
            tools.append(
                Tool(
                    name=func.name,
                    description=func.description,
                    func=func
                )
            )

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=customedPrefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        self.agent_memory = ConversationBufferMemory(memory_key="chat_history")

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain,
                              tools=tools, verbose=verbose
                              )
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools,
            verbose=verbose, memory=self.agent_memory,
            handle_parsing_errors="If the LLM output is a thought, stop calling any tools and respond using your own ability and knowledge as a chat AI. Otherwise, use the LLM output directly as your final answer!"   # Use the LLM output directly as your final answer!
        )

    def dialogue(self, input: str):
        print('TransGPT is running, Please wait for a moment...')
        with get_openai_callback() as cb:
            res = self.agent_chain.run(input=input, callbacks=[self.ch])
        # print('History: ', self.agent_memory.buffer)
        return res, cb
