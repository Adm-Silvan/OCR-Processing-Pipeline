import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configure Selenium Chrome WebDriver (headless optional)
chrome_options = Options()
# Uncomment for headless mode if you prefer no browser window
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def extract_info_from_url(url):
    driver.get(url)
    time.sleep(5)  # Wait for page to fully load
    
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"].strip() if meta_desc else ""
    
    # Initialize result variables
    author = ""
    translator = ""
    
    # Find all spans with class "hls-article-author-function"
    author_spans = soup.find_all("span", class_="hls-article-author-function")
    
    for span in author_spans:
        label = span.get_text(strip=True)
        next_node = span.next_sibling
        if next_node:
            text = next_node.strip()
            if label == "Autorin/Autor:":
                author = text
            elif label == "Übersetzung:":
                translator = text
    
    return description, author, translator

def main(input_csv, write_interval=5):
    df = pd.read_csv(input_csv)
    
    url_col = df.columns[5]  # Assuming URLs in 6th column
    
    if "description" not in df.columns:
        df["description"] = ""
    if "author" not in df.columns:
        df["author"] = ""
    if "translator" not in df.columns:
        df["translator"] = ""
    
    for i, url in enumerate(df[url_col]):
        try:
            desc, auth, trans = extract_info_from_url(url)
            df.at[i, "description"] = desc
            df.at[i, "author"] = auth
            df.at[i, "translator"] = trans
            print(f"Processed {i+1}/{len(df)}: {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")
        
        # Write to CSV every `write_interval` pages
        if (i + 1) % write_interval == 0:
            df.to_csv(input_csv, index=False)
            print(f"Intermediate save after {i+1} pages.")
    
    # Final write after loop completes
    df.to_csv(input_csv, index=False)
    print("Finished processing all URLs. Final save done.")

if __name__ == "__main__":
    input_csv_file = "GetData/Historisches_Lexikon/bios.csv"  # Change to your CSV filename
    main(input_csv_file, write_interval=5)
    driver.quit()
