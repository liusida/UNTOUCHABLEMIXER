"""
jarvis.py, Project 02 

Author: Kieran Edraney, Sida Liu, Sarah Smith, and Katherine Wilkinson <the Untouchable Mixer>
2020-04-05

Files required to run:
    botsettings.py
    jarvis.db

"""

# need do `pip install websocket-client`
import websocket

try:
    import thread
except ImportError:
    import _thread as thread
import time, urllib, json, sqlite3, os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier

import botsettings

msg_id = 1
is_training = 0
is_testing = 0
training_action_name = ""
db_connection = None
db_file = "jarvis.db"
JarvisBrain = None

def init():
    """ check all requirements """
    global db_connection
    if not os.path.exists(db_file):
        print(f"Database file {db_file} not found.")
        return
    try:
        db_connection = sqlite3.connect(db_file)
    except:
        print(f"Failed to open database file {db_file}.")
        return
    if 'enableTrace' not in dir(websocket):
        print(f"Wrong websocket package found. The websocket-client package should be installed by command:\npip install websocket-client")
        return


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
    c = db_connection.cursor()
    c.execute("INSERT INTO training_data (txt,action) VALUES (?, ?)", (text, action_name,))
    db_connection.commit() # save (commit) the changes

def read_all_training_text():
    """ read all training text in database and return a list """
    global db_connection, db_file
    c = db_connection.cursor()
    c.execute("SELECT * from training_data")
    rows = c.fetchall()
    data_x = []
    data_y = []
    for row in rows:
        data_x.append(row[0])
        data_y.append(row[1])
    return data_x, data_y

def build_a_brain():
    brain = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RidgeClassifier()),
        ])
    data_x, data_y = read_all_training_text()
    brain.fit(data_x, data_y)
    return brain


def on_message(ws, message):
    global is_training, is_testing, training_action_name, JarvisBrain

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

            if not is_training and not is_testing and msg_text=="training time":
                is_training = 1
                training_action_name = ""
                send_msg(ws, msg_channel, "OK, I'm ready for training. What NAME should this ACTION be?")
                return
            if not is_training and not is_testing and msg_text=="testing time":
                is_testing = 1
                send_msg(ws, msg_channel, "I'm training my brain with the data you've already given me...")
                JarvisBrain = build_a_brain()
                send_msg(ws, msg_channel, "OK, I'm ready for testing. Write me something and I'll try to figure it out.")
                return
            if not is_training and not is_testing and msg_text=="show data":
                data_x, data_y = read_all_training_text()
                s = "```"
                for i in range(len(data_x)):
                    s += f"({i:03}) [{data_y[i]:7}]: {data_x[i]} \n"
                s += "```"
                send_msg(ws, msg_channel, s)
                return

            # `done` has higher priority than those below
            if is_training and msg_text=="done":
                is_training = 0
                training_action_name = ""
                send_msg(ws, msg_channel, "OK, I'm finished training")
                return
            if is_testing and msg_text=="done":
                is_testing = 0
                training_action_name = ""
                send_msg(ws, msg_channel, "OK, I'm finished testing")
                return

            if is_training and training_action_name=="":
                training_action_name = msg_text.upper()
                send_msg(ws, msg_channel, "OK, let's call this action `{}`. Now give me some training text!".format(training_action_name))
                return
            
            if is_training and training_action_name!="":
                save_training_text(training_action_name, msg_text)
                send_msg(ws, msg_channel, "OK, I've got it! What else?")
                return

            if is_testing:
                if JarvisBrain is None:
                    JarvisBrain = build_a_brain()
                action = JarvisBrain.predict([msg_text])[0]
                send_msg(ws,msg_channel, f"OK, I think the action you mean is `{action}`...")
                send_msg(ws,msg_channel, "Write me something else and I'll try to figure it out.")
                return

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
    init()
    start()    
