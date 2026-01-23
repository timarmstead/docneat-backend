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

def parse_hsbc_logic(df):
    if df.empty: return pd.DataFrame()
    
    # 1. Identify key columns based on data patterns
    date_freq = {}
    num_freq = {}
    for _, row in df.iterrows():
        for i, v in enumerate(row.values):
            s = str(v).strip()
            if re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{2}', s):
                date_freq[i] = date_freq.get(i, 0) + 1
            # Check for numeric amount (ignoring currency symbols)
            cv = s.replace(',','').replace('£','').replace('$', '').strip()
            if cv and re.match(r'^-?\d*\.?\d+$', cv):
                num_freq[i] = num_freq.get(i, 0) + 1
                
    col_map = {'date': -1, 'out': -1, 'in': -1, 'bal': -1}
    if date_freq: col_map['date'] = max(date_freq, key=date_freq.get)
        
    # Valid numeric columns usually appear at the end of the row
    valid_nums = sorted([idx for idx, count in num_freq.items() if count > 0], reverse=True)
    
    if len(valid_nums) >= 1: col_map['bal'] = valid_nums[0]
    if len(valid_nums) >= 2: col_map['in'] = valid_nums[1]
    if len(valid_nums) >= 3: col_map['out'] = valid_nums[2]

    # Calculate where description likely starts and ends
    first_amt_col = min([c for c in [col_map['out'], col_map['in'], col_map['bal']] if c != -1] or [99])

    txns = []
    current_date = None
    
    for _, row in df.iterrows():
        # Convert row to list of strings
        vals = [str(v).strip() if v is not None and str(v).lower() != 'nan' else "" for v in row.values]
        if not any(vals): continue
        
        # Helper for safe index access (prevents the crash)
        def get_safe(idx):
            return vals[idx] if (idx != -1 and idx < len(vals)) else ""

        d_val = get_safe(col_map['date'])
        d_match = re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{2}', d_val)
        
        row_out = get_safe(col_map['out'])
        row_in = get_safe(col_map['in'])
        row_bal = get_safe(col_map['bal'])
        
        # Check if this row has any numeric data
        clean_out = row_out.replace(',','').replace('£','').strip()
        clean_in = row_in.replace(',','').replace('£','').strip()
        has_amt = any(re.match(r'^-?\d*\.?\d+$', x) for x in [clean_out, clean_in] if x)

        # Trigger new row if Date OR Amount found
        if d_match or has_amt:
            if d_match: current_date = d_match.group()
            
            # Description is everything between Date and Amounts
            desc_start = col_map['date'] + 1 if col_map['date'] != -1 else 0
            description = " ".join([v for v in vals[desc_start:min(first_amt_col, len(vals))] if v])
            
            txns.append({
                'Date': current_date,
                'Description': description,
                'Paid Out': clean_out if clean_out else "0",
                'Paid In': clean_in if clean_in else "0",
                'Balance': row_bal.replace(',','').replace('£','').strip()
            })
        elif txns:
            # Append orphaned text to last description
            extra = " ".join([v for i, v in enumerate(vals) if v and i < first_amt_col])
            if extra:
                txns[-1]['Description'] = (txns[-1]['Description'] + " " + extra).strip()

    return pd.DataFrame(txns)

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
        return JSONResponse(status_code=500, content={"error": f"Upload Failed: {str(e)}"})

@app.get("/status/{job_id}")
async def get_status(job_id: str, file_key: str = Query(...)):
    try:
        response = textract.get_document_analysis(JobId=job_id)
        if response['JobStatus'] == 'IN_PROGRESS': return {"status": "PROCESSING"}
        if response['JobStatus'] == 'FAILED': return {"status": "FAILED"}

        if response['JobStatus'] == 'SUCCEEDED':
            # 1. Gather all blocks from all pages
            all_blocks = response.get('Blocks', [])
            next_token = response.get('NextToken')
            while next_token:
                next_page = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                all_blocks.extend(next_page.get('Blocks', []))
                next_token = next_page.get('NextToken')
            
            bmap = {b['Id']: b for b in all_blocks}
            all_dfs = []
            
            # 2. Extract Tables
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
                        df_raw = pd.DataFrame.from_dict(grid, orient='index').sort_index(axis=1)
                        processed = parse_hsbc_logic(df_raw)
                        if not processed.empty: all_dfs.append(processed)

            # Cleanup S3
            try: s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
            except: pass

            if not all_dfs:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}
            
            # 3. Consolidate and Deduplicate
            final_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
            
            return {
                "status": "COMPLETED",
                "preview": final_df.head(100).to_dict(orient="records"),
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        # RETURN THE ACTUAL ERROR TO THE UI
        error_detail = traceback.format_exc()
        print(error_detail)
        return JSONResponse(status_code=500, content={"error": f"Logic Error: {str(e)}", "detail": error_detail})

@app.get("/")
def health(): return {"status": "V31 - Ragged Row Safety Active"}
