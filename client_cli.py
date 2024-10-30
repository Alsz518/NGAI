import logging
import argparse

from btrafficai.client import BaseChatBotClient

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname).1s:%(name)s] %(message)s'
)

default_logger = logging.getLogger('client_gr')


class ChatBotGradioClient(BaseChatBotClient):
    def start(self):
        default_logger.debug("Connecting...")
        result = self.zmq_rpc("start", client_id=self.client_id)
        default_logger.debug("Started: %s", result)

    @staticmethod
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

    def respond(self, message: str, chat_history: list):
        default_logger.debug("Sent: %s", message)
        response = self.zmq_rpc("respond", message=message)
        default_logger.debug("Received: %s", response)
        history = self.zmq_rpc("get_history")
        return "", self.convert_history(history)

    def reset(self):
        self.zmq_rpc("reset")

    def stop(self):
        self.zmq_rpc("stop")


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

    with BaseChatBotClient(args.server_url) as client:
        client.start()

        while True:
            try:
                request = input('> ')
            except (EOFError, InterruptedError) as ex:
                break
            response = client.zmq_rpc("respond", message=request)
            print(response)
            # history = client.zmq_rpc("get_history")
            # for row in history:
                # print(row)


if __name__ == '__main__':
    # 这个脚本仅测试用
    main()
