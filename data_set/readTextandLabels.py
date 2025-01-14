import pandas as pd
import os
import json


def read_labels(line):
    file_name, label_category_part = line.split('\t', 1)  # Split by tab to get file name and category-label-sublabel part
    category="" # URW or CC
    labels = []
    sublabels =[]

    
    if label_category_part == "Other\tOther":
        category="Other"
        labels.append("Other")
        sublabels.append("Other")
        return file_name, category, labels, sublabels
    
    # Split the label-category part by tab
    _, category_label_sublabel_pairs = label_category_part.split('\t', 1) 
    category_label_sublabel_pairs=category_label_sublabel_pairs.split(";")
    # Extract sublables, labels and categories from each pair - format category:label:sublabel  
    # it might happen that a label is listed more than once
    for pair in category_label_sublabel_pairs:
        if ':' in pair:
            category_, label_sublabel = pair.split(':', 1)
            category=category_
            label,sublabel = label_sublabel.split(':', 1)
            #if label.strip() not in labels: #if labels should be added only once
            labels.append(label.strip())
            sublabels.append(sublabel.strip())
    
    return file_name, category, labels, sublabels


def read_labels_single_language(file_path):
    # read labels
    all_file_names = []
    all_labels = []
    all_categories = []
    all_sublabels = []


    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            file_name, category, labels, sublabels = read_labels(line.strip())  # Process each line
            
            # Store the results
            all_file_names.append(file_name)  # Repeat the file name for each label
            all_labels.append(labels)
            all_categories.append(category)
            all_sublabels.append(sublabels)

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'File Name': all_file_names,
        'Category': all_categories,
        'Label': all_labels,
        'Sublabel':all_sublabels,
    })
    return df


'read labels of one language'
LANGUAGE="RU"
df_labels=read_labels_single_language(f"data_set/target_4_December_release/{LANGUAGE}/subtask-2-annotations.txt")



'read text of one language and save as JSON'
for index, row in df_labels.iterrows():
    
    file_name = row['File Name']
    category= row['Category']
    labels = row['Label']
    sublabels = row['Sublabel']

    content=""
    text_file_path=f'data_set/target_4_December_release/{LANGUAGE}/raw-documents/{file_name}'

    if os.path.exists(text_file_path):
        with open(text_file_path, 'r',encoding='utf-8') as file:
           content = file.read() 
    else:
        continue

    print(content)
    
   
    json_filename = f"{file_name.split('.')[0]}.json"  # Assuming you want to remove the original file extension
    save_path = f'data_set/target_4_December_JSON/{LANGUAGE}/'
    full_path = save_path+json_filename
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_to_save = {
        'text':content,
        'article_id': file_name,
        'category':category,
        'narratives': labels,
        'subnarratives': sublabels
    }

    
    with open(full_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)

