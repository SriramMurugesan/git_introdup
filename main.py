import json
import re

def convert_indian_numbers(value):
    match = re.match(r'(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*(lakhs?|crores?|L|Cr)', value, re.IGNORECASE)
    if match:
        num = float(match.group(1).replace(",", ""))
        unit = match.group(2).lower()
        if "lakh" in unit or "L" in unit:
            return num * 100000
        elif "crore" in unit or "Cr" in unit:
            return num * 10000000
    return float(value.replace(",", ""))

def extract_financial_entities(text):
    entities = []
    
    patterns = {
        'ORG': [
            r'[A-Z][a-zA-Z\s&]+(?:Ltd\.|Limited|Pvt\.|Private|Corporation|Company|Co\.|Group|Holdings|Technologies|Tech|Solutions|Industries|Enterprises)',
            r'(?:BSE|NSE):[A-Z]{1,10}\b',
            r'(?:NIFTY|SENSEX)(?:\s*50)?\b',
            r'\b(?:TCS|HDFC|SBI|ICICI|ONGC|ITC|L&T|M&M|BHEL|NTPC|SAIL)\b'
        ],
        'MONEY': [
            r'(?:\u20B9|Rs\.|INR|USD|\$|EUR|€|GBP|£)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr|millions?|billions?))?\b'
        ],
        'DATE': [
            r'\b(?:19|20)\d{2}\b',
            r'FY\d{2}(?:-\d{2})?\b',
            r'(?:Q[1-4]|H[12])\s*FY\d{2}(?:-\d{2})?\b',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}'
        ],
        'PERCENT': [
            r'\d+(?:\.\d+)?%',
            r'(?:increase|decrease|growth|decline|up|down|rose|fell|gained|lost|jumped|plunged)\s+(?:by\s+)?\d+(?:\.\d+)?%',
            r'\d+(?:\.\d+)?%\s+(?:YoY|QoQ|year-on-year|quarter-on-quarter)'
        ],
        'METRIC': [
            r'(?:P/E ratio|EPS|ROI|ROE|ROA|CAGR|margin|PAT|PBT|NPA|CASA ratio)\s*(?:of)?\s*\d+(?:\.\d+)?'
        ],
        'REGULATOR': [
            r'\b(?:RBI|SEBI|NSE|BSE|IRDAI|PFRDA|NABARD|SIDBI)\b',
            r'\b(?:Ministry of Finance|MoF|GST Council|Income Tax Department|IT Dept)\b'
        ]
    }
    
    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group().strip()
                if entity_type == 'MONEY':
                    matched_text = str(convert_indian_numbers(matched_text))
                score = 0.99 if entity_type in ['MONEY', 'PERCENT'] else 0.95
                entities.append({
                    "text": matched_text,
                    "entity": entity_type,
                    "score": score
                })
    
    unique_entities = {}
    for entity in entities:
        key = (entity['text'].lower(), entity['entity'])
        if key not in unique_entities or entity['score'] > unique_entities[key]['score']:
            unique_entities[key] = entity
    
    return sorted(list(unique_entities.values()), key=lambda x: x['score'], reverse=True)

def process_document(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    
    structured_data = extract_financial_entities(text)
    
    with open("output.json", "w") as output_file:
        json.dump(structured_data, output_file, indent=4)
    
    print("Extracted data saved to output.json")

if __name__ == "__main__":
    process_document("financial_report.txt")
