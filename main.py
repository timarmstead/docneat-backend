from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import re
import uuid
from pathlib import Path
import boto3
import os
import numpy as np
from typing import List, Dict

app = FastAPI()

# Standard CORS setup
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

textract = boto3.client(
    'textract',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def clean_bank_data(df):
    """Refined logic to handle multi-line transactions like Non-Sterling Fees."""
    if df.empty: return pd.DataFrame()
    
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    data_rows = []
    last_date = None
    
    # Identify money columns by looking for numeric content
    money_cols = []
    for c in range(df.shape[1]):
        col_str = "".join(df.iloc[:, c].astype(str)).replace(',', '')
        if any(char.isdigit() for char in col_str):
            money_cols.append(c)
    
    if len(money_cols) < 1: return pd.DataFrame()
    
    out_idx = money_cols[0]
    in_idx = money_cols[1] if len(money_cols) > 1 else out_idx
    bal_idx = money_cols[-1]

    for i in range(len(df)):
        row = df.iloc[i].astype(str).tolist()
        # Clean up 'nan' strings from Textract
        row = ["" if x.lower() == 'nan' else x for x in row]
        
        date_match = re.search(date_regex, row[0])
        if date_match:
            last_date = date_match.group()
        
        # Extract description (usually column 1 or 2)
        desc = row[1] if row[1].strip() else (row[2] if len(row) > 2 else "")
        
        # Extract and clean money
        p_out = row[out_idx].replace(',', '').replace('£', '').strip()
        p_in = row[in_idx].replace(',', '').replace('£', '').strip()
        bal = row[bal_idx].replace(',', '').replace('£', '').strip()

        if last_date and (desc.strip() or p_out or p_in):
            data_rows.append({
                'Date': last_date,
                'Description': desc.strip(),
                'Paid Out': p_out,
                'Paid In': p_in,
                'Balance': bal
            })

    # Merge multi-line descriptions (the 'Competitor' fix)
    final_rows = []
    for r in data_rows:
        # If no money on this line, append description to previous transaction
        if not r['Paid Out'] and not r['Paid In'] and final_rows and r['Date'] == final_rows[-1]['Date']:
            if r['Description']:
                final_rows[-1]['Description'] += " " + r['Description']
        else:
            final_rows.append(r)

    res_df = pd.DataFrame(final_rows)
    if not res_df.empty:
        for col in ['Paid Out', 'Paid In', 'Balance']:
            res_df[col] = pd.to_numeric(res_df[col], errors='coerce').fillna(0.0)
        # Filter out bank summary headers
        noise = ['BALANCE BROUGHT FORWARD', 'Account Summary', 'Sheet Number']
        res_df = res_df[~res_df['Description'].str.contains('|'.join(noise), case=False, na=False)]
    
    return res_df

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file.filename}"
    contents = await file.read()
    with open(input_path, "wb") as f: f.write(contents)

    all_dfs = []
    try:
        # 1. Get Textract Response
        response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
        blocks = response.get('Blocks', [])
        block_map = {b['Id']: b for b in blocks}
        
        # 2. Extract Tables
        for table_block in [b for b in blocks if b['BlockType'] == 'TABLE']:
            table_data = {}
            max_row, max_col = 0, 0
            for rel in table_block.get('Relationships', []):
                if rel['Type'] == 'CHILD':
                    for cell_id in rel['Ids']:
                        cell = block_map.get(cell_id)
                        if not cell: continue
                        r, c = cell['RowIndex'], cell['ColumnIndex']
                        max_row, max_col = max(max_row, r), max(max_col, c)
                        text = " ".join([block_map[w]['Text'] for rc in cell.get('Relationships', []) if rc['Type'] == 'CHILD' for w in rc['Ids'] if w in block_map])
                        table_data[(r, c)] = text.strip()
            
            grid = [[table_data.get((r, c), "") for c in range(1, max_col + 1)] for r in range(1, max_row + 1)]
            if grid:
                cleaned = clean_bank_data(pd.DataFrame(grid))
                if not cleaned.empty:
                    all_dfs.append(cleaned)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        # This prevents the 500 crash if AWS fails
        return {"error": "Processing failed", "details": str(e)}

    # 3. Combine and CRITICAL FIX for JSON Serialization
    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=['Date', 'Description', 'Paid Out', 'Paid In', 'Balance'])
    
    # Replace all NaN/Inf with None so JSON doesn't crash
    clean_preview = final_df.head(20).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
    
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    final_df.to_csv(csv_path, index=False)

    return {
        "preview": clean_preview,
        "csv_url": f"/download/{csv_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    return FileResponse(OUTPUT_DIR / name, media_type="text/csv", filename="docneat-export.csv")

@app.get("/")
def root(): return {"status": "DocNeat Logic V12 - Multi-Line & Serialization Fixed"}
