from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import re
import uuid
import boto3
import os
import numpy as np
from pathlib import Path

app = FastAPI()

# FULL CORS - This ensures the frontend and backend can talk without being blocked
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

def process_hsbc_logic(df):
    if df.empty: return pd.DataFrame()
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    txns = []
    last_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        
        # Date Logic
        d_match = re.search(date_regex, vals[0])
        if d_match: last_date = d_match.group()
        
        # Description (Col 1 or 2)
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        
        # Money (HSBC standard cols)
        out_v = vals[3].replace(',','').replace('£','') if len(vals) > 3 else ""
        in_v = vals[4].replace(',','').replace('£','') if len(vals) > 4 else ""
        bal_v = vals[5].replace(',','').replace('£','') if len(vals) > 5 else ""

        if last_date and (desc or out_v or in_v):
            is_new = any(c.isdigit() for c in (out_v + in_v))
            if not is_new and txns and last_date == txns[-1]['Date']:
                txns[-1]['Description'] += " " + desc
            else:
                txns.append({
                    'Date': last_date,
                    'Description': desc,
                    'Paid Out': out_v if out_v else "0",
                    'Paid In': in_v if in_v else "0",
                    'Balance': bal_v if bal_v else "0"
                })
    return pd.DataFrame(txns)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    try:
        # Read file
        contents = await file.read()
        
        # 1. AWS Textract
        response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
        
        # 2. Extract Tables
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
            all_dfs.append(process_hsbc_logic(df))

        if not all_dfs:
            return JSONResponse(status_code=400, content={"error": "No tables detected"})

        final_df = pd.concat(all_dfs, ignore_index=True)
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        # 3. JSON Safe Conversion (Fixes the NaN/Inf crash)
        preview = final_df.head(20).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        
        csv_filename = f"{file_id}.csv"
        csv_path = OUTPUT_DIR / csv_filename
        final_df.to_csv(csv_path, index=False)

        # Note: Frontend expects 'csv_url' to look like this
        return {
            "preview": preview,
            "csv_url": f"/download/{csv_filename}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Fixed Route for Download
@app.get("/download/{name}")
async def download(name: str):
    file_path = OUTPUT_DIR / name
    if file_path.exists():
        return FileResponse(file_path, media_type="text/csv", filename="bank_statement.csv")
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/")
def root():
    return {"status": "DocNeat Live", "engine": "V17"}
