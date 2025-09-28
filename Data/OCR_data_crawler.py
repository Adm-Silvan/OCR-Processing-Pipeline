from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import re
import time
import os
import json
import datetime

def tablescrape(year,month):
    # Set up Firefox WebDriver in headless mode
    options = Options()
    options.add_argument("--headless")  # Remove this line if you want to see the browser

    driver = webdriver.Firefox(options=options)

    try:
        url = f"https://www.chgov.bar.admin.ch/browser/?year={year}&month={month}"
        driver.get(url)

        # Wait for the table to appear (adjust selector if needed)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".aiii-timeline-results__inner table"))
        )

        # Let JavaScript render everything
        time.sleep(2)  # Sometimes needed for dynamic content

        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.select_one(".aiii-timeline-results__inner table")

        result = {}

        if table:
            for row in table.find_all("tr")[1:]:  # Skip header
                cells = row.find_all("td")
                if len(cells) >= 2:
                    date = cells[0].get_text(strip=True)
                    link = cells[1].find("a")
                    signature = cells[2].get_text(strip=True)
                    if link and link.has_attr("href"):
                        match = re.search(r'/(\d+)\.json', link["href"])
                        if match:
                            number = match.group(1)
                            result[date] = [number,signature]
        manifest = f"C:/Data/OCR_Protocols/Manifests/{year}-{month+1}.json"
        with open(manifest, 'w') as json_file:
            json.dump(result, json_file, indent=4)
        return(result)
    finally:
        driver.quit()

def download_ocr(table):
    error_log = "C:/Data/OCR_Protocols/error_log.txt"
    for document in list(table):
        doc_id = table.get(document)[0]
        signature = table.get(document)[1]
        url = f"https://api.chgov.bar.admin.ch//ocr/{doc_id}/{doc_id}.txt"
        year = document.split(".")[-1]
        month = document.split(".")[-2]
        day = document.split(".")[-3]
        doc_date = f"{year}-{month}-{day}"
        path= f"C:/Data/OCR_Protocols/{year}/{month}"
        file = path+f"/{doc_date}.txt"
        if not os.path.exists(path):
            os.makedirs(path)
        response = requests.get(url)
        if response.status_code == 200:
            with open(file, "wb") as f:
                f.write(response.content)
            print(f"File downloaded successfully to {file}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            with open(error_log,"+a") as log:
                log.write(f"{datetime.datetime.now()}: Failed to download {doc_date}.txt    ID:{doc_id}     Signature: {signature}    Error: {response.status_code}\n")


for year in range(1917,1974):
    for month in range(0,12):
        table = tablescrape(year,month)
        download_ocr(table)





