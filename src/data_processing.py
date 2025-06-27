import json
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def parse_punishment_years(text_description: str) -> Tuple[Optional[int], str]:
    """
    Parses punishment years from a text description.
    
    Args:
        text_description: The text containing punishment details.
    
    Returns:
        Tuple containing the number of years (or None) and the punishment text.
    """
    years = None
    punishment_text = ""
    
    punishment_match = re.search(r'Punishment:\s*(.*)', text_description, re.IGNORECASE)
    if punishment_match:
        punishment_text = punishment_match.group(1).strip()
    
    year_ranges = re.findall(r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years|year)', punishment_text, re.IGNORECASE)
    if year_ranges:
        years = max(int(yr) for r in year_ranges for yr in r)
    else:
        year_matches = re.findall(r'(\d+)\s*(?:years|year)', punishment_text, re.IGNORECASE)
        if year_matches:
            years = int(year_matches[0])
    
    if years is None:
        if 'life' in punishment_text.lower():
            years = 999
        elif 'death' in punishment_text.lower():
            years = 1000
    
    return years, punishment_text

def load_data_structured(file_path: str) -> List[Dict]:
    """
    Loads and structures IPC data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file.
    
    Returns:
        List of dictionaries containing structured IPC data.
    """
    structured_data = []
    skipped_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    text_content = entry.get('text', '')
                    
                    if "nan" in text_content.lower() or len(text_content) < 20:
                        skipped_count += 1
                        continue
                    
                    ipc_section_match = re.search(r'IPC\s*(\d+[A-Z]*)', text_content, re.IGNORECASE)
                    ipc_section_id = ipc_section_match.group(0).upper() if ipc_section_match else "UNKNOWN"
                    
                    punishment_years, punishment_description_text = parse_punishment_years(text_content)
                    
                    structured_data.append({
                        "ipc_section_id": ipc_section_id,
                        "description_summary": text_content.split(':', 1)[1].split('Punishment:', 1)[0].strip() if 'Punishment:' in text_content else text_content,
                        "punishment_years": punishment_years,
                        "original_text": text_content
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    logger.warning(f"Error parsing line: {e}")
        logger.info(f"Loaded {len(structured_data)} IPC sections, skipped {skipped_count} invalid entries.")
    except FileNotFoundError:
        logger.error(f"Data file {file_path} not found.")
        raise
    return structured_data

