import csv
csv_path = "bios.csv"
with open(csv_path, mode='r', newline='', encoding="utf-8") as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)
    for row in reader:
        print(row)