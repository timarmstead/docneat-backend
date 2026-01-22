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

# Use /tmp as it's the only writable directory on Render
UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Minimal AWS Client
textract = boto3.client(
    'textract',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def parse_hsbc_rows(df):
    """Refined parser to stop skipping multi-line fees."""
    if df.empty: return pd.DataFrame()
    
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    data = []
    curr_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        # Look for date in first col
        if re.search(date_regex, vals[0]):
            curr_date = re.search(date_regex, vals[0]).group()
            
        # Description is column 1 or 2
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        
        # Money check: Paid Out(3), Paid In(4), Balance(5)
        p_out = vals[3].replace(',','').replace('£','') if len(vals) > 3 else ""
        p_in = vals[4].replace(',','').replace('£','') if len(vals) > 4 else ""
        bal = vals[5].replace(',','').replace('£','') if len(vals) > 5 else ""

        if curr_date and (desc or p_out or p_in):
            has_money = any(c.isdigit() for c in (p_out + p_in))
            if not has_money and data and curr_date == data[-1]['Date']:
                data[-1]['Description'] += " " + desc
            else:
                data.append({
                    'Date': curr_date,
                    'Description': desc,
                    'Paid Out': p_out if p_out else "0",
                    'Paid In': p_in if p_in else "0",
                    'Balance': bal if bal else "0"
                })
    return pd.DataFrame(data)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    try:
        contents = await file.read()
        
        # 1. Textract Call
        response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
        
        # 2. Extract Blocks
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        all_dfs = []
        
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = txt
            
            df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            all_dfs.append(parse_hsbc_rows(df))

        if not all_dfs:
            return {"error": "No tables found"}

        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Clean numeric cols
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        # 3. JSON Safe Export
        preview = final_df.head(25).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        csv_path = OUTPUT_DIR / f"{file_id}.csv"
        final_df.to_csv(csv_path, index=False)

        return {"preview": preview, "csv_url": f"/download/{file_id}.csv"}

    except Exception as e:
        return {"error": "Processing failed", "details": str(e)}

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(OUTPUT_DIR / filename, media_type="text/csv", filename="bank_export.csv")

@app.get("/")
def health(): return {"status": "DocNeat V16 - Core Engine Only"}
