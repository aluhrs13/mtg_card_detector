import json
from collections import Counter

# Load the JSON data
with open('./_data/json/unique-artwork-20241206100306.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Filter the data to include only cards with "frame" value of 2003 or 2015
filtered_data = [card for card in data if card.get('frame') in ["2003", "2015"]]
data = filtered_data

# Extract the "frame" values
frames = [card['layout'] for card in data]

# Count the unique "frame" values
frame_counts = Counter(frames)

# Print the unique "frame" values and their counts
for frame, count in frame_counts.items():
    print(f'layout: {frame}, Count: {count}')
