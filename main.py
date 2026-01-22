from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import re
import boto3
import os
import numpy as np
import io

app = FastAPI()

# Robust CORS for browser-server communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

textract = boto3.client(
    'textract',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def clean_hsbc_logic(df):
    """Specific parser for HSBC transaction rows [cite: 1, 354, 371]"""
    if df.empty: return pd.DataFrame()
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    transactions = []
    current_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        d_match = re.search(date_regex, vals[0])
        if d_match: current_date = d_match.group()
        
        # Column mapping based on standard HSBC PDF layout [cite: 1, 13]
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        p_out = vals[3].replace(',','').replace('£','') if len(vals) > 3 else "0"
        p_in = vals[4].replace(',','').replace('£','') if len(vals) > 4 else "0"
        bal = vals[5].replace(',','').replace('£','') if len(vals) > 5 else "0"

        if current_date and (desc or p_out != "0" or p_in != "0"):
            # Multi-line handling for 'Non-Sterling' fees [cite: 361, 362]
            is_new_txn = any(c.isdigit() for c in (p_out + p_in))
            if not is_new_txn and transactions and current_date == transactions[-1]['Date']:
                transactions[-1]['Description'] += " " + desc
            else:
                transactions.append({
                    'Date': current_date, 'Description': desc,
                    'Paid Out': p_out, 'Paid In': p_in, 'Balance': bal
                })
    return pd.DataFrame(transactions)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Read into memory directly
        file_bytes = await file.read()
        
        # 1. AWS Textract processing
        response = textract.analyze_document(Document={'Bytes': file_bytes}, FeatureTypes=['TABLES'])
        
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        dfs = []
        
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    text = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = text
            
            raw_df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            dfs.append(clean_hsbc_logic(raw_df))

        if not dfs:
            return JSONResponse(status_code=400, content={"error": "No data found in PDF"})

        final_df = pd.concat(dfs, ignore_index=True)
        
        # 2. Cleanup and JSON safety formatting [cite: 371]
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)
        
        preview_data = final_df.head(20).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")

        # 3. Create CSV in memory (No disk writing = no 500 errors)
        stream = io.StringIO()
        final_df.to_csv(stream, index=False)
        
        return {
            "preview": preview_data,
            "csv_content": stream.getvalue() # Send the actual CSV data to the frontend
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def health(): return {"status": "DocNeat V18 Engine Active"}
