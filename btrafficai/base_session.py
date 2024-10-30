
class BaseUserSession:
    def __init__(self):
        self.started = False
        self.client_id = None
        # chat history already sent to client
        self.chat_history_sent = []
        # new chat history not seen by client
        self.chat_history_new = []

    def start(self, client_id: str, **kwargs):
        self.client_id = client_id
        self.clear_history()
        self.started = True

    def stop(self):
        pass

    def restart(self):
        self.stop()
        self.start(self.client_id)

    def respond(self, message: str):
        sent = {"dir": 1, "type": "text", "msg": message}
        received = {"dir": 2, "type": "text", "msg": "Received: " + message}
        result = [sent, received]
        self.chat_history_new.extend(result)
        return result

    def set_history(self, chat_history):
        self.chat_history_sent = list(chat_history)
        self.chat_history_new.clear()

    def get_history(self, full=False):
        new_history = self.chat_history_new
        self.chat_history_sent.extend(new_history)
        self.chat_history_new = []
        if full:
            return self.chat_history_sent
        else:
            return new_history

    def clear_history(self):
        self.chat_history_sent.clear()
        self.chat_history_new.clear()

    def reset(self):
        self.clear_history()
