import json
import os
from pathlib import Path

# Standard SQuAD format keys - only these will be kept when combining datasets
SQUAD_KEYS = ['id', 'title', 'context', 'question', 'answers']


def combine_jsonl_files(file_paths, output_file, keep_only_squad_keys=True):
    """
    Combine multiple JSONL files into a single JSONL file.
    
    Args:
        file_paths (list): List of paths to JSONL files to combine
        output_file (str): Path to the output combined JSONL file
        keep_only_squad_keys (bool): If True, only keep standard SQuAD keys
    
    Returns:
        int: Total number of rows written to the output file
    """
    total_rows = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"⚠️  Warning: File {file_path} does not exist. Skipping...")
                continue
            
            print(f"📂 Processing: {file_path}")
            file_rows = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                # Validate JSON
                                json_obj = json.loads(line)
                                
                                # Filter to only SQuAD keys if requested
                                if keep_only_squad_keys:
                                    json_obj = {key: json_obj[key] for key in SQUAD_KEYS if key in json_obj}
                                
                                # Write to output
                                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                                file_rows += 1
                            except json.JSONDecodeError as e:
                                print(f"❌ Error parsing JSON in {file_path} at line {line_num}: {e}")
                                continue
                
                print(f"   ✅ Added {file_rows} rows from {file_path}")
                total_rows += file_rows
                
            except Exception as e:
                print(f"❌ Error reading file {file_path}: {e}")
                continue
    
    print(f"\n🎉 Successfully combined {len(file_paths)} files into {output_file}")
    print(f"📊 Total rows written: {total_rows}")
    if keep_only_squad_keys:
        print(f"🔑 Kept only SQuAD keys: {SQUAD_KEYS}")
    
    return total_rows


def combine_jsonl_with_metadata(file_paths, output_file, add_source=True, keep_only_squad_keys=True):
    """
    Combine multiple JSONL files with optional source metadata.
    
    Args:
        file_paths (list): List of paths to JSONL files to combine
        output_file (str): Path to the output combined JSONL file
        add_source (bool): Whether to add source file information to each row
        keep_only_squad_keys (bool): If True, only keep standard SQuAD keys (before adding metadata)
    
    Returns:
        int: Total number of rows written to the output file
    """
    total_rows = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"⚠️  Warning: File {file_path} does not exist. Skipping...")
                continue
            
            source_name = Path(file_path).stem  # Get filename without extension
            print(f"📂 Processing: {file_path}")
            file_rows = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                json_obj = json.loads(line)
                                
                                # Filter to only SQuAD keys if requested
                                if keep_only_squad_keys:
                                    json_obj = {key: json_obj[key] for key in SQUAD_KEYS if key in json_obj}
                                
                                # Add source metadata if requested
                                if add_source:
                                    json_obj['_source_file'] = source_name
                                    json_obj['_source_path'] = file_path
                                
                                # Write modified JSON to output
                                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                                file_rows += 1
                            except json.JSONDecodeError as e:
                                print(f"❌ Error parsing JSON in {file_path} at line {line_num}: {e}")
                                continue
                
                print(f"   ✅ Added {file_rows} rows from {file_path}")
                total_rows += file_rows
                
            except Exception as e:
                print(f"❌ Error reading file {file_path}: {e}")
                continue
    
    print(f"\n🎉 Successfully combined {len(file_paths)} files into {output_file}")
    print(f"📊 Total rows written: {total_rows}")
    if keep_only_squad_keys:
        print(f"🔑 Kept only SQuAD keys: {SQUAD_KEYS}")
    
    return total_rows


if __name__ == "__main__":
    # Example usage
    file_list = [
        "../Datasets/adversarial_qa_train.jsonl",
        "../Datasets/checklist_train.jsonl",
        "../Datasets/squad_train.jsonl"
    ]
    
    output_path = "../Datasets/combined_train.jsonl"
    
    # Simple combination
    total_rows = combine_jsonl_files(file_list, output_path)
    
    # Or with metadata
    # total_rows = combine_jsonl_with_metadata(file_list, output_path, add_source=True)
