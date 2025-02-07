import os
import json
from pathlib import Path

def deduplicate_labels(input_directory, output_directory, language):
    """
    Process JSON files to remove duplicate narratives and subnarratives within each sample
    and save to a new directory.
    
    Args:
        input_directory (str): Path to input directory containing JSON files
        output_directory (str): Path to output directory for cleaned JSON files
        language (str): Language code ('EN' or 'PT') for logging purposes
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics for reporting
    total_files = 0
    files_with_duplicates = 0
    total_duplicates_removed = 0
    
    # Process each JSON file
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            total_files += 1
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename)
            
            with open(input_filepath, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    original_data = data.copy()
                    
                    # Remove duplicates from narratives if present
                    if 'narratives' in data:
                        original_narratives_count = len(data['narratives'])
                        data['narratives'] = list(dict.fromkeys(data['narratives']))
                        narratives_removed = original_narratives_count - len(data['narratives'])
                    else:
                        narratives_removed = 0
                    
                    # Remove duplicates from subnarratives if present
                    if 'subnarratives' in data:
                        original_subnarratives_count = len(data['subnarratives'])
                        data['subnarratives'] = list(dict.fromkeys(data['subnarratives']))
                        subnarratives_removed = original_subnarratives_count - len(data['subnarratives'])
                    else:
                        subnarratives_removed = 0
                    
                    # Update statistics
                    total_removed = narratives_removed + subnarratives_removed
                    if total_removed > 0:
                        files_with_duplicates += 1
                        total_duplicates_removed += total_removed
                    
                    # Save cleaned data
                    with open(output_filepath, 'w', encoding='utf-8') as outfile:
                        json.dump(data, outfile, indent=2, ensure_ascii=False)
                    
                except json.JSONDecodeError:
                    print(f"Error decoding {filename}")
                    continue
    
    # Create and save summary report
    summary = {
        'language': language,
        'total_files_processed': total_files,
        'files_with_duplicates': files_with_duplicates,
        'total_duplicates_removed': total_duplicates_removed,
        'percentage_files_with_duplicates': round((files_with_duplicates / total_files * 100), 2) if total_files > 0 else 0
    }
    
    with open(os.path.join(output_directory, f'deduplication_summary_{language}.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def process_both_languages(base_input_dir, base_output_dir):
    """
    Process both English and Portuguese datasets.
    
    Args:
        base_input_dir (str): Base directory containing language subdirectories
        base_output_dir (str): Base directory for output
    """
    # Process English data
    en_input = os.path.join(base_input_dir, 'EN')
    en_output = os.path.join(base_output_dir, 'EN_cleaned')
    en_summary = deduplicate_labels(en_input, en_output, 'EN')
    
    # Process Portuguese data
    pt_input = os.path.join(base_input_dir, 'PT')
    pt_output = os.path.join(base_output_dir, 'PT_cleaned')
    pt_summary = deduplicate_labels(pt_input, pt_output, 'PT')
    
    # Print summaries
    print("\nEnglish Dataset Summary:")
    print(f"Total files processed: {en_summary['total_files_processed']}")
    print(f"Files with duplicates: {en_summary['files_with_duplicates']}")
    print(f"Total duplicates removed: {en_summary['total_duplicates_removed']}")
    print(f"Percentage of files with duplicates: {en_summary['percentage_files_with_duplicates']}%")
    
    print("\nPortuguese Dataset Summary:")
    print(f"Total files processed: {pt_summary['total_files_processed']}")
    print(f"Files with duplicates: {pt_summary['files_with_duplicates']}")
    print(f"Total duplicates removed: {pt_summary['total_duplicates_removed']}")
    print(f"Percentage of files with duplicates: {pt_summary['percentage_files_with_duplicates']}%")

# Example usage
base_input_directory = 'data_set/target_4_December_JSON'
base_output_directory = 'data_set/cleaned_data'

process_both_languages(base_input_directory, base_output_directory)