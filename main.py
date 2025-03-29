from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

def extract_financial_entities(text):
    entities = []
    
    # Enhanced patterns for Indian financial entities
    patterns = {
        'ORG': [
            r'[A-Z][a-zA-Z\s&]+(?:Ltd\.|Limited|Pvt\.|Private|Corporation|Company|Co\.|Group|Holdings|Technologies|Tech|Solutions|Industries|Enterprises)',
            r'(?:BSE|NSE):[A-Z]{1,10}\b',
            r'(?:NIFTY|SENSEX)(?:\s*50)?\b',
            r'\b(?:TCS|HDFC|SBI|ICICI|ONGC|ITC|L&T|M&M|BHEL|NTPC|SAIL)\b'
        ],
        'MONEY': [
            r'(?:\₹|Rs\.|INR|Rupees?)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr|k|mn|bn))?\b',
            r'\d+(?:,\d{2,3})*(?:\.\d+)?\s*(?:lakhs?|crores?|L|Cr)\b',
            r'(?:revenue|profit|loss|earnings|EBITDA|income|debt|assets|liabilities|turnover)\s+of\s+(?:\₹|Rs\.|INR)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr))?\b'
        ],
        'DATE': [
            r'\b(?:19|20)\d{2}\b',
            r'FY\d{2}(?:-\d{2})?\b',
            r'(?:Q[1-4]|H[12])\s*FY\d{2}(?:-\d{2})?\b',
            r'(?:Q[1-4]|first quarter|second quarter|third quarter|fourth quarter)(?:\s+of)?\s+(?:19|20)\d{2}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}'
        ],
        'PERCENT': [
            r'\d+(?:\.\d+)?%',
            r'(?:increase|decrease|growth|decline|up|down|rose|fell|gained|lost|jumped|plunged)\s+(?:by\s+)?\d+(?:\.\d+)?%',
            r'\d+(?:\.\d+)?%\s+(?:YoY|QoQ|year-on-year|quarter-on-quarter)',
            r'\d+(?:\.\d+)?\s+(?:percentage points|basis points|bps)'
        ],
        'METRIC': [
            r'(?:P/E ratio|EPS|ROI|ROE|ROA|CAGR|margin|PAT|PBT|NPA|CASA ratio)\s*(?:of)?\s*\d+(?:\.\d+)?',
            r'(?:market cap|market value|valuation|mcap)\s+of\s+(?:\₹|Rs\.|INR)\s*\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|L|Cr))?\b',
            r'(?:NPA|CASA|CRR|SLR|PLR|NBFC)\s*(?:ratio|rate)?\s*(?:of)?\s*\d+(?:\.\d+)?%?'
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
                score = 0.99 if entity_type in ['MONEY', 'PERCENT'] else 0.95
                
                if not any(e['text'].lower() == matched_text.lower() and e['entity'] == entity_type for e in entities):
                    entities.append({
                        "text": matched_text,
                        "entity": entity_type,
                        "score": score
                    })
    
    return sorted(entities, key=lambda x: x['score'], reverse=True)

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
