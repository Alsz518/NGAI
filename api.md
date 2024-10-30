# REST API
Response format: JSON
* code: HTTP status code
* client_id: The session ID
* data

## ChatBot
### POST /chatbot/start
Request params: None

Response:
* client_id

### POST /chatbot/respond
Chat with the bot.

Request params:
* client_id
* message: text

Response:
* client_id
* data: array of
  * dir: 1, user; 2, bot
  * type: text/img
  * msg

### POST /chatbot/stop
Stop the user session.

Request params:
* client_id

### POST /chatbot/reset
Reset the bot state.

Request params:
* client_id

### POST /chatbot/get_history
Request params:
* client_id
* full (optional): 0 or 1, get full history or last messages

Response:
* client_id
* data: array of
  * dir: 1, user; 2, bot
  * type: text/img
  * msg
