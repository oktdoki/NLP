import os
import json
import random
from collections import Counter
import numpy as np
from pathlib import Path

def get_category_distribution(directory):
    """Get the distribution of subnarratives in a directory."""
    subnarratives_count = Counter()
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    if 'subnarratives' in data:
                        subnarratives_count.update(data['subnarratives'])
                except json.JSONDecodeError:
                    continue
    
    return subnarratives_count

def balance_dataset(input_directory, output_directory, target_other_size=100, target_minority_size=50):
    """
    Balance the dataset using a hybrid approach of undersampling majority class
    and oversampling minority classes.
    """
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Read all files and group by subnarratives
    files_by_subnarrative = {}
    all_files = []
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    file_data = {'filename': filename, 'data': data}
                    all_files.append(file_data)
                    
                    if 'subnarratives' in data:
                        for subnarrative in data['subnarratives']:
                            if subnarrative not in files_by_subnarrative:
                                files_by_subnarrative[subnarrative] = []
                            files_by_subnarrative[subnarrative].append(file_data)
                except json.JSONDecodeError:
                    continue

    # Separate "Other" and non-"Other" categories
    other_files = files_by_subnarrative.get("Other", [])
    non_other_categories = {k: v for k, v in files_by_subnarrative.items() if k != "Other"}

    # Undersample "Other" category
    if len(other_files) > target_other_size:
        sampled_other_files = random.sample(other_files, target_other_size)
    else:
        sampled_other_files = other_files

    # Process each category
    processed_files = set()
    balanced_files = []

    # Add undersampled "Other" files
    for file_data in sampled_other_files:
        if file_data['filename'] not in processed_files:
            balanced_files.append(file_data)
            processed_files.add(file_data['filename'])

    # Process non-"Other" categories
    for category, files in non_other_categories.items():
        if len(files) < target_minority_size:
            # Oversample if needed
            oversampled_files = files
            while len(oversampled_files) < target_minority_size:
                additional = random.choice(files)
                oversampled_files.append({
                    'filename': f"synthetic_{len(oversampled_files)}_{additional['filename']}",
                    'data': additional['data'].copy()
                })
        else:
            # Undersample if needed
            oversampled_files = random.sample(files, target_minority_size)

        # Add to balanced dataset
        for file_data in oversampled_files:
            if file_data['filename'] not in processed_files:
                balanced_files.append(file_data)
                processed_files.add(file_data['filename'])

    # Save balanced dataset
    for file_data in balanced_files:
        output_path = os.path.join(output_directory, file_data['filename'])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_data['data'], f, indent=2, ensure_ascii=False)

    # Generate and save statistics
    original_stats = get_category_distribution(input_directory)
    balanced_stats = get_category_distribution(output_directory)
    
    stats = {
        'original_distribution': dict(original_stats),
        'balanced_distribution': dict(balanced_stats),
        'total_original_files': len(all_files),
        'total_balanced_files': len(balanced_files)
    }
    
    with open(os.path.join(output_directory, 'balance_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def balance_both_languages(base_input_dir, base_output_dir):
    """Balance both English and Portuguese datasets."""
    # Process English data
    en_input = os.path.join(base_input_dir, 'EN_cleaned')
    en_output = os.path.join(base_output_dir, 'EN_balanced')
    en_stats = balance_dataset(en_input, en_output)
    
    # Process Portuguese data
    pt_input = os.path.join(base_input_dir, 'PT_cleaned')
    pt_output = os.path.join(base_output_dir, 'PT_balanced')
    pt_stats = balance_dataset(pt_input, pt_output)
    
    print("\nEnglish Dataset Balance Statistics:")
    print(f"Original total files: {en_stats['total_original_files']}")
    print(f"Balanced total files: {en_stats['total_balanced_files']}")
    print("\nPortuguese Dataset Balance Statistics:")
    print(f"Original total files: {pt_stats['total_original_files']}")
    print(f"Balanced total files: {pt_stats['total_balanced_files']}")

# Example usage
base_input_directory = 'data_set/cleaned_data'
base_output_directory = 'data_set/balanced_data'
balance_both_languages(base_input_directory, base_output_directory)