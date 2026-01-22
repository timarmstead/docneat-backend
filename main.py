import io
import os
import re
import uuid
import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

textract = boto3.client('textract',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

BUCKET_NAME = os.getenv('AWS_S3_BUCKET')

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
            if any(c.isdigit() for c in (p_out + p_in)):
                txns.append({'Date': current_date, 'Description': desc, 'Paid Out': p_out or "0", 'Paid In': p_in or "0", 'Balance': bal or ""})
            elif txns:
                txns[-1]['Description'] += " " + desc
    return pd.DataFrame(txns)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """STEP 1: Uploads to S3 and starts Textract. Returns JobId immediately."""
    file_key = f"uploads/{uuid.uuid4()}-{file.filename}"
    try:
        content = await file.read()
        s3.put_object(Bucket=BUCKET_NAME, Key=file_key, Body=content)
        
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': BUCKET_NAME, 'Name': file_key}},
            FeatureTypes=['TABLES']
        )
        # We return the JobId and the FileKey so the frontend can check status
        return {"job_id": response['JobId'], "file_key": file_key}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{job_id}")
async def get_status(job_id: str, file_key: str):
    """STEP 2: Called by the frontend to check if the PDF is done."""
    try:
        response = textract.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        
        if status == 'IN_PROGRESS':
            return {"status": "PROCESSING"}
        
        if status == 'FAILED':
            return {"status": "FAILED"}

        if status == 'SUCCEEDED':
            # Collect all pages of results
            pages = [response]
            next_token = response.get('NextToken')
            while next_token:
                next_page = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                pages.append(next_page)
                next_token = next_page.get('NextToken')
            
            # Parse all tables
            all_dfs = []
            for pg in pages:
                blocks = pg.get('Blocks', [])
                bmap = {b['Id']: b for b in blocks}
                for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
                    grid = {}
                    for rel in table.get('Relationships', []):
                        for cid in rel['Ids']:
                            cell = bmap[cid]
                            r, c = cell['RowIndex'], cell['ColumnIndex']
                            txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids'] if bmap[w]['BlockType'] == 'WORD'])
                            grid.setdefault(r, {})[c] = txt
                    df_raw = pd.DataFrame.from_dict(grid, orient='index').sort_index(axis=1)
                    all_dfs.append(parse_hsbc_logic(df_raw))

            # Immediate Cleanup
            s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)

            if not all_dfs:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}

            final_df = pd.concat(all_dfs, ignore_index=True)
            for col in ['Paid Out', 'Paid In', 'Balance']:
                final_df[col] = pd.to_numeric(final_df[col].replace('', '0'), errors='coerce').fillna(0.0)

            return {
                "status": "COMPLETED",
                "preview": final_df.head(100).to_dict(orient="records"),
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def health():
    return {"status": "V26 Multi-Endpoint Active"}
