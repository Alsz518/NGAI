import logging
import argparse

from btrafficai.client import BaseChatBotClient
import gradio as gr

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname).1s:%(name)s] %(message)s'
)

default_logger = logging.getLogger('client_gr')


class ChatBotGradioClient(BaseChatBotClient):
    def respond(self, message: str, chat_history: list):
        default_logger.debug("Sent: %s", message)
        response = self.zmq_rpc("respond", message=message)
        default_logger.debug("Received: %s", response)
        history = self.zmq_rpc("get_history", full=True)
        return "", self.convert_history(history)

    def reset(self):
        self.zmq_rpc("reset")

    def stop(self):
        self.zmq_rpc("stop")


def convert_history(chat_history: list):
    result = []
    for row in chat_history:
        content = row['msg'] if row['type'] == 'text' else (row['msg'],)
        if row['dir'] == 1:
            result.append([content, None])
        elif result and result[-1][1] is None:
            result[-1][1] = content
        else:
            result.append([None, content])
    return result


def main():
    parser = argparse.ArgumentParser(description='User session manager')
    parser.add_argument(
        "-s", "--server-url", default="tcp://127.0.0.1:7890",
        help="command server url")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show debug log")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    server_url = args.server_url
    with ChatBotGradioClient(server_url) as client:
        client_id = client.client_id
        client.start()

    def respond(message: str, chat_history: list):
        with ChatBotGradioClient(server_url, client_id=client_id) as client:
            default_logger.debug("Sent: %s", message)
            response = client.cmd_respond(message=message)
            default_logger.debug("Received: %s", response)
            history = client.cmd_get_history(full=True)
        return "", convert_history(history)

    def reset():
        with ChatBotGradioClient(server_url, client_id=client_id) as client:
            client.cmd_reset()

    with gr.Blocks(analytics_enabled=False) as demo:
        chatbot = gr.Chatbot(min_width=500, height=800)
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(reset)
    demo.launch(width=600)


if __name__ == '__main__':
    # 这个脚本仅测试用
    main()
