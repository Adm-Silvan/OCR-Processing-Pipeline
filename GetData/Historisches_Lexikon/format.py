import csv
import re

input_file = "liste_bio_d_utf8.csv"
output_file = "bios.csv"

def extract_first_4digit_number(s):
    """
    Extract all numbers from string s.
    Take the first number found.
    If number longer than 4 digits, take first 4 digits.
    Return (number_string, slash_removed)
    slash_removed is True if '/' character found and removed from s.
    """
    slash_removed = '/' in s
    # Extract all digit sequences
    numbers = re.findall(r'\d+', s)
    if not numbers:
        return "", slash_removed
    # Take first number only
    first_num = numbers[0]
    # If longer than 4 digits, truncate
    if len(first_num) > 4:
        first_num = first_num[:4]
    return first_num, slash_removed

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Ensure row length for safety
        while len(row) < 8:
            row.append("")

        # Split column 4 on dash
        parts = row[3].split('-', 1)
        part1 = parts[0].strip()
        part2 = parts[1].strip() if len(parts) > 1 else ""

        # Extract number and uncertainty for part1
        num1, uncertain1 = extract_first_4digit_number(part1)
        # Extract number and uncertainty for part2
        num2, uncertain2 = extract_first_4digit_number(part2)

        # Set values in columns
        row[3] = num1
        row[4] = num2 if num2 else ""

        # Set uncertainty flags in columns 7 and 8
        row[6] = "uncertain" if uncertain1 else ""
        row[7] = "uncertain" if uncertain2 else ""

        writer.writerow(row)

