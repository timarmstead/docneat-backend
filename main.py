import io
import os
import re
import uuid
import boto3
import traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query
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

def get_clean_amt(val):
    if not val: return ""
    s = str(val).replace('£','').replace('$','').replace(',','').strip()
    # Matches numbers like 1234.56 or -1234.56
    match = re.search(r'-?\d+\.\d{2}', s)
    return match.group(0) if match else ""

def parse_hsbc_logic(df, sticky_date):
    if df.empty: return [], sticky_date
    
    # 1. Detect Columns (Date and Numeric)
    col_map = {'date': -1, 'out': -1, 'in': -1, 'bal': -1}
    num_counts = {}
    for _, row in df.iterrows():
        for i, v in enumerate(row.values):
            s = str(v).strip()
            if re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{2}', s):
                col_map['date'] = i
            if get_clean_amt(s):
                num_counts[i] = num_counts.get(i, 0) + 1
                
    # Balance is usually the last column with numbers
    sorted_nums = sorted([i for i, c in num_counts.items() if c > 0], reverse=True)
    if len(sorted_nums) >= 1: col_map['bal'] = sorted_nums[0]
    if len(sorted_nums) >= 2: col_map['in'] = sorted_nums[1]
    if len(sorted_nums) >= 3: col_map['out'] = sorted_nums[2]

    first_amt_idx = min([c for c in [col_map['out'], col_map['in'], col_map['bal']] if c != -1] or [len(df.columns)])
    
    # Summary block filtering
    blacklist = ["opening balance", "closing balance", "payments in", "payments out", "payment type and details", "how we protect"]
    
    txns = []
    
    for _, row in df.iterrows():
        vals = [str(v).strip() if v is not None and str(v).lower() != 'nan' else "" for v in row.values]
        row_str = " ".join(vals).lower()
        
        # Filter summary noise and legal footers
        if any(item in row_str for item in blacklist) and not "forward" in row_str:
            continue
        if len(row_str) > 400: # Catch giant legal footer blocks
            continue

        # Date Detection
        d_idx = col_map['date']
        d_val = vals[d_idx] if d_idx != -1 else ""
        d_match = re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{2}', d_val)
        if d_match: sticky_date = d_match.group()
        
        # Amount Detection
        p_out = get_clean_amt(vals[col_map['out']]) if col_map['out'] != -1 else ""
        p_in = get_clean_amt(vals[col_map['in']]) if col_map['in'] != -1 else ""
        p_bal = get_clean_amt(vals[col_map['bal']]) if col_map['bal'] != -1 else ""
        
        has_money = p_out or p_in or p_bal
        
        # Determine Description
        desc_start = col_map['date'] + 1 if col_map['date'] != -1 else 0
        desc = " ".join([v for v in vals[desc_start:first_amt_idx] if v])

        # Logic: New line if Date OR Money found
        if d_match or has_money:
            # Handle Brought/Carried Forward rows specifically
            if "forward" in desc.lower():
                final_bal = p_bal or p_in or p_out # Ensure balance is caught even if column shifted
                txns.append({'Date': sticky_date, 'Description': desc, 'Paid Out': '', 'Paid In': '', 'Balance': final_bal})
            else:
                txns.append({'Date': sticky_date, 'Description': desc, 'Paid Out': p_out, 'Paid In': p_in, 'Balance': p_bal})
        elif txns and desc:
            # Append orphaned text to previous transaction
            txns[-1]['Description'] = (txns[-1]['Description'] + " " + desc).strip()

    return txns, sticky_date

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    clean_name = re.sub(r'[^a-zA-Z0-9.]', '_', file.filename)
    file_key = f"uploads/{uuid.uuid4()}-{clean_name}"
    try:
        content = await file.read()
        s3.put_object(Bucket=BUCKET_NAME, Key=file_key, Body=content)
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': BUCKET_NAME, 'Name': file_key}},
            FeatureTypes=['TABLES']
        )
        return {"job_id": response['JobId'], "file_key": file_key}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{job_id}")
async def get_status(job_id: str, file_key: str = Query(...)):
    try:
        response = textract.get_document_analysis(JobId=job_id)
        if response['JobStatus'] == 'IN_PROGRESS': return {"status": "PROCESSING"}
        if response['JobStatus'] == 'FAILED': return {"status": "FAILED"}

        if response['JobStatus'] == 'SUCCEEDED':
            all_blocks = response.get('Blocks', [])
            next_token = response.get('NextToken')
            while next_token:
                next_page = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                all_blocks.extend(next_page.get('Blocks', []))
                next_token = next_page.get('NextToken')
            
            bmap = {b['Id']: b for b in all_blocks}
            final_data = []
            sticky_date = "" # Global date memory across all tables
            
            # Group blocks by table
            for block in all_blocks:
                if block['BlockType'] == 'TABLE':
                    grid = {}
                    if 'Relationships' not in block: continue
                    for rel in block['Relationships']:
                        for cell_id in rel['Ids']:
                            cell = bmap.get(cell_id)
                            if not cell or 'RowIndex' not in cell: continue
                            r, c = cell['RowIndex'], cell['ColumnIndex']
                            text = ""
                            if 'Relationships' in cell:
                                for child_rel in cell['Relationships']:
                                    for word_id in child_rel['Ids']:
                                        word_block = bmap.get(word_id)
                                        if word_block: text += word_block.get('Text', '') + " "
                            grid.setdefault(r, {})[c] = text.strip()
                    
                    if grid:
                        df_table = pd.DataFrame.from_dict(grid, orient='index').sort_index(axis=1)
                        # Process table and update the global sticky_date
                        table_txns, sticky_date = parse_hsbc_logic(df_table, sticky_date)
                        final_data.extend(table_txns)

            try: s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
            except: pass

            if not final_data:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}
            
            final_df = pd.DataFrame(final_data).drop_duplicates()
            # Absolute cleanup for JSON safety
            final_df = final_df.astype(str).replace(['nan', 'None', 'NaN', '0.00'], '')
            
            return {
                "status": "COMPLETED",
                "preview": final_df.head(100).to_dict(orient="records"),
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Logic Error: {str(e)}", "detail": traceback.format_exc()})

@app.get("/")
def health(): return {"status": "V39 - Robust Global Context Active"}
