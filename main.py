from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
import boto3
import os
import numpy as np
import io
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We need S3 for Asynchronous processing (AWS requirement for multi-page)
# If you don't have an S3 bucket yet, we can try to force the sync call to 'look' at all pages
# but for multi-page PDFs, AWS officially requires an S3 bucket.
# Let's try the "Multi-Page Sync" hack first:

def get_textract_client():
    return boto3.client(
        'textract',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def parse_hsbc_logic(df):
    if df.empty: return pd.DataFrame()
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    txns = []
    current_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        d_match = re.search(date_regex, vals[0])
        if d_match: current_date = d_match.group()
            
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        p_out = vals[3].replace(',','').replace('£','').strip() if len(vals) > 3 else ""
        p_in = vals[4].replace(',','').replace('£','').strip() if len(vals) > 4 else ""
        bal = vals[5].replace(',','').replace('£','').strip() if len(vals) > 5 else ""

        if current_date and (desc or p_out or p_in):
            has_money = any(c.isdigit() for c in (p_out + p_in))
            if has_money:
                txns.append({
                    'Date': current_date, 'Description': desc,
                    'Paid Out': p_out if p_out else "0",
                    'Paid In': p_in if p_in else "0",
                    'Balance': bal if bal else ""
                })
            elif txns:
                txns[-1]['Description'] += " " + desc
    return pd.DataFrame(txns)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        client = get_textract_client()
        
        # New: Use 'FeatureTypes' with 'QUERIES' or just 'TABLES' 
        # but specifically handling the byte stream for multi-page
        response = client.analyze_document(
            Document={'Bytes': file_bytes},
            FeatureTypes=['TABLES']
        )
        
        # ... (rest of the mapping logic remains same as V23)
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        dfs = []
        
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = txt
            
            df_table = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            dfs.append(parse_hsbc_logic(df_table))

        if not dfs:
            return {"preview": [{"Date": "N/A", "Description": "No tables found", "Paid Out": 0, "Paid In": 0, "Balance": 0}], "csv_content": ""}

        final_df = pd.concat(dfs, ignore_index=True)
        preview = final_df.to_dict(orient="records")
        stream = io.StringIO()
        final_df.to_csv(stream, index=False)
        
        return {"preview": preview, "csv_content": stream.getvalue()}

    except Exception as e:
        # If it's a multi-page issue, let's suggest the fix
        return {"preview": [{"Date": "DIAGNOSTIC", "Description": f"Error: {str(e)}", "Paid Out": 0, "Paid In": 0, "Balance": 0}], "csv_content": "Error"}
