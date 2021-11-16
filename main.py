from mqtt import Mqtt
from gdrive import Drive
from image_processor import ImageProcessor
import json
drive = Drive()
messenger = Mqtt("server", address='broker.emqx.io', port=1883)
image_processor= ImageProcessor()
def detect(drive_link : str) -> int:
    frame = Drive.download(drive_link) #ini udah opencv object
    rows, cols, channels = frame.shape
    ImageProcessor.detect()
    
def process_notification(message : str):
    data = json.loads(message)
    total_violations = detect(data['link'])
    if total_violations:
        notify_server(data,total_violations)

def notify_server(data : dict, total_violations : int):
    payload = {
        "class" : data['class'],
        "number_of_violations" : total_violations,
        "photo_link" : data['link']
    }
    messenger.publish(json.dumps(payload), '3deef803-2854-495d-b641-677c7bda1979')
    return


def main():
    messenger.subscribe("59288f20-4e6d-4423-938a-84b9dcfc7be4", process_notification)
    messenger.start()
    while 1:
        continue



if __name__ == '__main__':
    main()