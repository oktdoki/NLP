import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def truncate_label(label, max_words=3):
    """Truncate label to first max_words words with ellipsis, except for 'Other'."""
    if label == "Other":
        return label
    
    words = label.split()
    return ' '.join(words[:max_words]) + '...'

def analyze_subnarratives_multilingual(en_directory, pt_directory):
    # Counters for both languages
    en_subnarratives_count = Counter()
    pt_subnarratives_count = Counter()
    
    # Process English files
    for filename in os.listdir(en_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(en_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    if 'subnarratives' in data:
                        filtered_subnarratives = [
                            subnarrative for subnarrative in data['subnarratives'] 
                            if not subnarrative.startswith("Discrediting Ukrainian government")
                        ]
                        en_subnarratives_count.update(filtered_subnarratives)
                except json.JSONDecodeError:
                    print(f"Error decoding {filename}")
    
    # Process Portuguese files
    for filename in os.listdir(pt_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(pt_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    if 'subnarratives' in data:
                        filtered_subnarratives = [
                            subnarrative for subnarrative in data['subnarratives'] 
                            if not subnarrative.startswith("Discrediting Ukrainian government")
                        ]
                        pt_subnarratives_count.update(filtered_subnarratives)
                except json.JSONDecodeError:
                    print(f"Error decoding {filename}")

    # Get top 6 subnarratives for each language
    top_en = en_subnarratives_count.most_common(6)
    top_pt = pt_subnarratives_count.most_common(6)

    # Prepare data for plotting
    en_labels = [truncate_label(item[0]) for item in top_en]
    en_counts = [item[1] for item in top_en]
    pt_labels = [truncate_label(item[0]) for item in top_pt]
    pt_counts = [item[1] for item in top_pt]

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Set width of bars and positions of the bars
    width = 0.35
    x = np.arange(max(len(top_en), len(top_pt)))

    # Create bars
    rects1 = ax.bar(x - width/2, en_counts, width, label='English', color='skyblue')
    rects2 = ax.bar(x + width/2, pt_counts, width, label='Portuguese', color='lightcoral')

    # Customize the plot
    ax.set_title('Top 6 Subnarratives by Language', pad=20)
    ax.set_xlabel('Subnarratives (First 3 Words)')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(en_labels, rotation=45, ha='right')

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # Add legend
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig('multilingual_subnarratives_balanced.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Save results to a JSON file
    results_dict = {
        'english': {truncate_label(k): v for k, v in top_en},
        'portuguese': {truncate_label(k): v for k, v in top_pt}
    }
    
    with open('multilingual_subnarratives_results_balanced.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)

    return results_dict

# Example usage
en_directory = 'data_set/balanced_data/EN_balanced'
pt_directory = 'data_set/balanced_data/PT_balanced'
results = analyze_subnarratives_multilingual(en_directory, pt_directory)
print(results)