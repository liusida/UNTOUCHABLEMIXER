# need do `pip install websocket-client`
import websocket

try:
    import thread
except ImportError:
    import _thread as thread
import time
import urllib
import urllib.request
import json
import sqlite3

import botsettings

msg_id = 1
is_training = 0
is_testing = 0
training_action_name = ""
db_connection = None
db_file = "jarvis.db"

def start_rtm():
    URL = "https://slack.com/api/rtm.connect?token={}".format(
        botsettings.bot_jarvis)  # try change this to bot_sarah
    connection = urllib.request.urlopen(URL)
    text = connection.read().decode("utf-8")
    data = json.loads(text)
    ws_url = data["url"]
    return ws_url

def send_msg(ws, channel, message):
    global msg_id
    reply_obj = {"id": msg_id, "type": "message", "channel": channel, "text": message}
    msg_id += 1
    reply_str = json.dumps(reply_obj)
    ws.send(reply_str)

def save_training_text(action_name, text):
    global db_connection, db_file
    if db_connection is None:
        db_connection = sqlite3.connect(db_file)
    c = db_connection.cursor()
    c.execute("INSERT INTO training_data (txt,action) VALUES (?, ?)", (text, action_name,))
    db_connection.commit() # save (commit) the changes

def on_message(ws, message):
    global is_training, is_testing, training_action_name

    try:
        msg_data = json.loads(message)
    except:
        print("received a broken message:")
        return
    print(message)
    if "type" in msg_data:
        if msg_data["type"] == "message":
            print("="*10, "DEBUG", "="*10)
            print("Received a message:")
            print("Channel: ", msg_data["channel"])
            print("Message: ", msg_data["text"])
            msg_text = msg_data["text"]
            msg_channel = msg_data["channel"]
            if not is_training and not is_testing and msg_text=="testing time":
                pass
            if not is_training and msg_text=="training time":
                is_training = 1
                training_action_name = ""
                send_msg(ws, msg_channel, "OK, I'm ready for training. What NAME should this ACTION be?")
                return
            
            # `done` has higher priority than those below
            if is_training and msg_text=="done":
                is_training = 0
                training_action_name = ""
                send_msg(ws, msg_channel, "OK, I'm finished training")
                return

            if is_training and training_action_name=="":
                training_action_name = msg_text.upper()
                send_msg(ws, msg_channel, "OK, let's call this action `{}`. Now give me some training text!".format(training_action_name))
                return
            
            if is_training and training_action_name!="":
                save_training_text(training_action_name, msg_text)
                send_msg(ws, msg_channel, "OK, I've got it! What else?")
            
        if msg_data["type"] == "hello":
            print("I am online now. Go to Slack and send me a direct message!")


def on_error(ws, error):
    print("ERROR:", error)


def on_close(ws):
    if db_connection is not None:
        db_connection.close()
    print("### closed ###")

def start():
    ws_url = start_rtm()

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

if __name__ == "__main__":
    start()    
