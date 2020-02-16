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
        botsettings.bot_sida)
    print(URL)
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
    if "type" in msg_data and msg_data["type"]=="message":
        print("="*10)
        print("this is a message:")
        print(msg_data["text"])
        print("="*10)

    # ws.send("hi")

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        for i in range(3):
            time.sleep(1)
            ws.send("Hello %d" % i)
        time.sleep(1)
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


if __name__ == "__main__":
    ws_url = start_rtm()

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(ws_url,
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    # ws.on_open = on_open
    ws.run_forever()

