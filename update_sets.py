import json
from collections import Counter, defaultdict
from datetime import datetime

# Load the JSON data from the file
with open('./_data/json/sets.json', 'r') as file:
    data = json.load(file)
data = data['data']

# Extract and count the "set_type" values and total number of cards
set_type_counts = Counter()
card_counts = defaultdict(int)

for item in data:
    #released_at = item.get('released_at')
    #if released_at and datetime.strptime(released_at, '%Y-%m-%d').year > 2010:
    set_type = item['set_type']

    if set_type in ['expansion', 'commander', 'masters', 'draft_innovation', 'core', 'masterpiece']:
        set_type_counts[set_type] += 1
        card_counts[set_type] += item.get('card_count', 0)

# Sort the results by card count
sorted_results = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)

# Print the results
for set_type, card_count in sorted_results:
    print(f"{set_type}: {set_type_counts[set_type]} sets, {card_count} cards")

# Probably only care about ['expansion', 'commander', 'masters', 'draft_innovation', 'core', 'masterpiece']