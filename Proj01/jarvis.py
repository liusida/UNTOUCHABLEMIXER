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

import botsettings


def start_rtm():
    URL = "https://slack.com/api/rtm.connect?token={}".format(
        botsettings.bot_sida)  # try change this to bot_sarah
    connection = urllib.request.urlopen(URL)
    text = connection.read().decode("utf-8")
    data = json.loads(text)
    ws_url = data["url"]
    return ws_url


def on_message(ws, message):
    try:
        msg_data = json.loads(message)
    except:
        print("received a broken message:")
        print(message)
        return
    if "type" in msg_data:
        if msg_data["type"] == "message":
            print("="*10)
            print("this is a message:")
            print(msg_data["text"])
            print("="*10)
        if msg_data["type"] == "hello":
            print("I am online now. Go to Slack and send me a direct message!")


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


if __name__ == "__main__":
    ws_url = start_rtm()

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
