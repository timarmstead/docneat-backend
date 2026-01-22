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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DIAGNOSTIC: Check if Env Vars exist at startup
AWS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REG = os.getenv('AWS_REGION', 'us-east-1')

def get_textract_client():
    return boto3.client(
        'textract',
        region_name=AWS_REG,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )

def clean_hsbc_logic(df):
    if df.empty: return pd.DataFrame()
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    txns = []
    last_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        d_match = re.search(date_regex, vals[0])
        if d_match: last_date = d_match.group()
        
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        p_out = vals[3].replace(',','').replace('£','') if len(vals) > 3 else "0"
        p_in = vals[4].replace(',','').replace('£','') if len(vals) > 4 else "0"
        bal = vals[5].replace(',','').replace('£','') if len(vals) > 5 else "0"

        if last_date and (desc or p_out != "0" or p_in != "0"):
            is_new = any(c.isdigit() for c in (p_out + p_in))
            if not is_new and txns and last_date == txns[-1]['Date']:
                txns[-1]['Description'] += " " + desc
            else:
                txns.append({'Date': last_date, 'Description': desc, 'Paid Out': p_out, 'Paid In': p_in, 'Balance': bal})
    return pd.DataFrame(txns)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not AWS_KEY or not AWS_SECRET:
        return JSONResponse(status_code=500, content={"error": "AWS Credentials Missing in Render Env Vars"})

    try:
        file_bytes = await file.read()
        client = get_textract_client()
        
        # 1. AWS Call with explicit error catching
        try:
            response = client.analyze_document(Document={'Bytes': file_bytes}, FeatureTypes=['TABLES'])
        except Exception as aws_err:
            return JSONResponse(status_code=500, content={"error": f"AWS Textract Failed: {str(aws_err)}"})
        
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
            
            df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            dfs.append(clean_hsbc_logic(df))

        if not dfs:
            return JSONResponse(status_code=400, content={"error": "No tables detected in PDF"})

        final_df = pd.concat(dfs, ignore_index=True)
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)
        
        preview = final_df.head(25).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        
        stream = io.StringIO()
        final_df.to_csv(stream, index=False)
        
        return {
            "preview": preview,
            "csv_content": stream.getvalue()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"General Server Error: {str(e)}"})

@app.get("/")
def health():
    return {"status": "DocNeat V19", "aws_configured": bool(AWS_KEY)}
