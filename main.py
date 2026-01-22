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

textract = boto3.client(
    'textract',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def process_table_data(df):
    """
    State-machine parser for HSBC multi-line statements.
    Ensures 'Non-Sterling Fees' and sub-descriptions are matched to the correct transaction.
    """
    if df.empty: return pd.DataFrame()
    
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    transactions = []
    current_date = None
    
    # Identify key columns based on content
    # Column 0: Date, Column 1/2: Description, Column 3/4/5: Money
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        line_text = " ".join(vals)
        
        # Filter noise
        if any(x in line_text for x in ["Account Summary", "Sheet Number", "HBUKGB"]): continue
        
        # Update date if found
        date_match = re.search(date_regex, vals[0])
        if date_match:
            current_date = date_match.group()
            
        # Extract Money - HSBC Standard: Out(3), In(4), Balance(5)
        # We strip symbols to check if the string is numeric
        p_out = vals[3].replace(',','').replace('£','').strip() if len(vals) > 3 else ""
        p_in = vals[4].replace(',','').replace('£','').strip() if len(vals) > 4 else ""
        bal = vals[5].replace(',','').replace('£','').strip() if len(vals) > 5 else ""
        
        # Description is usually in col 1 or 2
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        
        if current_date and (desc or p_out or p_in):
            # Check if this is a continuation line (no money on this line)
            is_money = any(c.isdigit() for c in (p_out + p_in))
            
            if not is_money and transactions and current_date == transactions[-1]['Date']:
                # Append to previous description (e.g. "Non-Sterling Fee")
                transactions[-1]['Description'] += " " + desc
            else:
                # New transaction line
                transactions.append({
                    'Date': current_date,
                    'Description': desc,
                    'Paid Out': p_out if p_out else "0",
                    'Paid In': p_in if p_in else "0",
                    'Balance': bal if bal else "0"
                })

    return pd.DataFrame(transactions)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    try:
        contents = await file.read()
        
        # 1. AWS Textract Call
        response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
        
        # 2. Map Textract Blocks to DataFrames
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        
        all_dfs = []
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    # Get cell text
                    txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = txt
            
            df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            all_dfs.append(process_table_data(df))

        if not all_dfs:
            return {"error": "No tables found"}

        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Numerical conversion
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        # 3. CRITICAL: JSON-safe formatting to prevent frontend crash
        preview = final_df.head(20).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        
        csv_path = OUTPUT_DIR / f"{file_id}.csv"
        final_df.to_csv(csv_path, index=False)

        return {
            "preview": preview,
            "csv_url": f"/download/{file_id}.csv"
        }

    except Exception as e:
        print(f"System Error: {str(e)}")
        return {"error": "Processing Error", "details": str(e)}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(OUTPUT_DIR / filename, media_type="text/csv", filename="bank_export.csv")

@app.get("/")
def health(): return {"status": "DocNeat Light-Engine V15 Active"}
