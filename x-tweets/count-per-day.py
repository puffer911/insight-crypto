import os
import re
from collections import Counter

# Buat daftar nama file: 11.txt sampai 30.txt
files = [f"{i}.txt" for i in range(11, 31)]

# Counter untuk jumlah tweet per hari
date_counter = Counter()

# Regex tanggal ISO (2025-04-XX)
date_regex = re.compile(r'\d{4}-\d{2}-\d{2}')

# Proses tiap file
for filename in files:
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        continue

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    tweets = content.strip().split('\n\n')  # asumsi pemisah antar tweet

    for tweet in tweets:
        match = date_regex.search(tweet)
        if match:
            date = match.group()
            date_counter[date] += 1

# Tampilkan hasil terurut
for date in sorted(date_counter):
    print(f"{date}: {date_counter[date]} tweet")
