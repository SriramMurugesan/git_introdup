from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

def extract_financial_entities(text):
    entities = []
    
    # Enhanced patterns for Indian financial entities
    patterns = {
        'ORG': [
            # Indian companies with common suffixes
            r'[A-Z][a-zA-Z\s&]+(?:Ltd\.|Limited|Pvt\.|Private|Corporation|Company|Co\.|Group|Holdings|Technologies|Tech|Solutions|Industries|Enterprises)',
            # BSE/NSE stock symbols
            r'(?:BSE|NSE):[A-Z]{1,10}\b',  # e.g., BSE:RELIANCE, NSE:TCS
            r'(?:NIFTY|SENSEX)(?:\s*50)?\b',
            # Common Indian company abbreviations
            r'\b(?:TCS|HDFC|SBI|ICICI|ONGC|ITC|L&T|M&M|BHEL|NTPC|SAIL)\b'
        ],
        'MONEY': [
            # Rupee amounts with symbols and words
            r'(?:₹|Rs\.|INR|Rupees?)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr|k|mn|bn))?\b',
            # Amounts in lakhs/crores
            r'\d+(?:,\d{2,3})*(?:\.\d+)?\s*(?:lakhs?|crores?|L|Cr)\b',
            # Revenue/profit amounts in Indian format
            r'(?:revenue|profit|loss|earnings|EBITDA|income|debt|assets|liabilities|turnover)\s+of\s+(?:₹|Rs\.|INR)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr))?\b'
        ],
        'DATE': [
            # Years
            r'\b(?:19|20)\d{2}\b',
            # Indian fiscal year format
            r'FY\d{2}(?:-\d{2})?\b',  # e.g., FY23, FY23-24
            r'(?:Q[1-4]|H[12])\s*FY\d{2}(?:-\d{2})?\b',  # e.g., Q1 FY23
            # Quarters with years
            r'(?:Q[1-4]|first quarter|second quarter|third quarter|fourth quarter)(?:\s+of)?\s+(?:19|20)\d{2}',
            # Indian months
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}'
        ],
        'PERCENT': [
            # Percentage values
            r'\d+(?:\.\d+)?%',
            # Growth/decline rates
            r'(?:increase|decrease|growth|decline|up|down|rose|fell|gained|lost|jumped|plunged)\s+(?:by\s+)?\d+(?:\.\d+)?%',
            # YoY and QoQ growth
            r'\d+(?:\.\d+)?%\s+(?:YoY|QoQ|year-on-year|quarter-on-quarter)',
            # Percentage points and basis points
            r'\d+(?:\.\d+)?\s+(?:percentage points|basis points|bps)'
        ],
        'METRIC': [
            # Indian financial metrics
            r'(?:P/E ratio|EPS|ROI|ROE|ROA|CAGR|margin|PAT|PBT|NPA|CASA ratio)\s*(?:of)?\s*\d+(?:\.\d+)?',
            # Market terms
            r'(?:market cap|market value|valuation|mcap)\s+of\s+(?:₹|Rs\.|INR)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr))?\b',
            # Indian banking terms
            r'(?:NPA|CASA|CRR|SLR|PLR|NBFC)\s*(?:ratio|rate)?\s*(?:of)?\s*\d+(?:\.\d+)?%?'
        ],
        'REGULATOR': [
            # Indian regulatory bodies and exchanges
            r'\b(?:RBI|SEBI|NSE|BSE|IRDAI|PFRDA|NABARD|SIDBI)\b',
            # Government bodies
            r'\b(?:Ministry of Finance|MoF|GST Council|Income Tax Department|IT Dept)\b'
        ]
    }
    
    # Process each pattern category
    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the matched text and clean it up
                matched_text = match.group().strip()
                # Assign confidence scores based on pattern reliability
                score = 0.99 if entity_type in ['MONEY', 'PERCENT'] else 0.95
                
                entities.append({
                    "text": matched_text,
                    "entity": entity_type,
                    "score": score
                })
    
    # Remove duplicates while keeping the highest score
    unique_entities = {}
    for entity in entities:
        key = (entity['text'].lower(), entity['entity'])
        if key not in unique_entities or entity['score'] > unique_entities[key]['score']:
            unique_entities[key] = entity
    
    return sorted(list(unique_entities.values()), key=lambda x: x['score'], reverse=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    results = extract_financial_entities(text)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
