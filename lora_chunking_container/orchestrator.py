import os
import json
import signal
import requests
import time

PATH = "C:/Users/Silvan/Data/OCR_Protocols/"
def handler(signum, frame):
    global terminate
    print("Termination signal received, exiting gracefully.")
    terminate = True

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
# load the JSON file
terminate = False
terminate_count = 0

url = "http://localhost:7000/process"

for year in range(1876,1904):
    if terminate: break
    for month in range(1,13):
        with open(PATH+f"Manifests/{year}-{month}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        for date, values in data.items():
            id = values[0]
            signature = values[1]
            print(f"Date: {date}, ID: {id},Signature {signature}")
            day = date.split(".")[0]
            if len(str(month)) < 2: month = "0"+str(month)
            if os.path.exists(PATH+f"{year}/{month}/{year}-{month}-{day}.txt"):
                with open(PATH+f"{year}/{month}/{year}-{month}-{day}.txt", "r", encoding="utf-8") as text_file:
                    text = text_file.read()
                data = {
                    "text": text,
                    "id": id,
                    "signature": signature,
                    "date": f"{year}-{month}-{day}.txt"
                }
                while True:
                    response = requests.post(url, json=data)
                    if response.status_code == 200:
                        print("Success:", response.json())
                        break
                    elif response.status_code == 503:
                        print("Still processing...")
                        time.sleep(3)
                    else:
                        print(f"Error {response.status_code}")
                        break
            else:
                print(PATH+f"{year}/{month}/{year}-{month}-{day}.txt is missing")
                with open("missing_files.txt", "a") as file:
                    file.write(date + " " + values[0]+ " " + values[1] + "\n")




"""             if terminate:
                if terminate_count == 0:
                    end_year = year
                decision = input("received termination signal. Finish data for the year? y/n ")
                if "y" in decision:
                    print("Will continue to finish data for this year")
                    if year != end_year:
                        break
                    else:
                        continue
                elif "n" in decision:
                    print("Will terminate immediately")                    
                    break """
                    