from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import re
import uuid
import boto3
import os
import numpy as np
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize AWS with simpler config
textract = boto3.client(
    'textract',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def parse_hsbc_table(df):
    """The 'Competitor-Grade' parser: ensures sub-lines like fees are captured."""
    if df.empty: return pd.DataFrame()
    
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    clean_rows = []
    current_date = None
    
    for _, row in df.iterrows():
        # Convert row to list of strings and strip whitespace
        vals = [str(v).strip() for v in row.values]
        line_text = " ".join(vals)
        
        # Skip header/footer noise
        if any(x in line_text for x in ["Account Summary", "Sheet Number", "HBUKGB"]): continue
        
        # Look for date
        date_found = re.search(date_regex, vals[0])
        if date_found:
            current_date = date_found.group()
        
        # Dynamically find money (last 3 columns usually)
        # We filter out empty strings and 'nan'
        money_vals = []
        for v in vals[1:]:
            clean_v = v.replace(',', '').replace('£', '')
            if clean_v and any(c.isdigit() for c in clean_v) and not re.search(date_regex, v):
                money_vals.append(clean_v)

        # Description is usually in the middle
        desc = vals[1] if len(vals) > 1 else ""
        
        if current_date and (desc or money_vals):
            clean_rows.append({
                'Date': current_date,
                'Description': desc,
                'MoneyRaw': money_vals,
                'FullRow': vals
            })

    # Grouping Logic: Merge lines that belong together
    final_transactions = []
    for entry in clean_rows:
        # If this line has no money but the previous one did, it might be a sub-description
        # OR if this line HAS money but NO date, it's a new transaction on the same date
        is_new_txn = any(any(c.isdigit() for c in m) for m in entry['MoneyRaw'])
        
        if not is_new_txn and final_transactions:
            final_transactions[-1]['Description'] += " " + entry['Description']
        else:
            # Try to map MoneyRaw to Out/In/Balance
            m = entry['MoneyRaw']
            final_transactions.append({
                'Date': entry['Date'],
                'Description': entry['Description'],
                'Paid Out': m[0] if len(m) == 3 else (m[0] if len(m) == 1 else "0"),
                'Paid In': m[1] if len(m) == 3 else "0",
                'Balance': m[-1] if len(m) > 0 else "0"
            })

    return pd.DataFrame(final_transactions)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    try:
        contents = await file.read()
        
        # Call Textract
        response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
        
        # Map blocks
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        
        all_data = []
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = txt
            
            df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            all_data.append(parse_hsbc_table(df))

        if not all_data:
            return {"error": "No transaction tables found in document."}

        final_df = pd.concat(all_data, ignore_index=True)
        
        # Final formatting
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        # JSON Safe Preview
        preview = final_df.head(15).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        
        csv_path = OUTPUT_DIR / f"{file_id}.csv"
        final_df.to_csv(csv_path, index=False)

        return {"preview": preview, "csv_url": f"/download/{file_id}.csv"}

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return {"error": "Internal Server Error", "message": str(e)}

@app.get("/download/{file_id}.csv")
async def download(file_id: str):
    return FileResponse(OUTPUT_DIR / f"{file_id}.csv", media_type="text/csv", filename="statement_export.csv")
