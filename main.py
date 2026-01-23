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

def process_all_rows(all_rows):
    """
    Processes a flat list of rows from the entire document.
    Ensures dates propagate across page/table boundaries.
    """
    if not all_rows: return pd.DataFrame()

    # 1. Column detection based on global data patterns
    # We find the most likely indices for Date, Out, In, and Balance
    date_freq = {}
    num_freq = {}
    for row in all_rows:
        for i, v in enumerate(row):
            s = str(v).strip()
            if re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{2}', s):
                date_freq[i] = date_freq.get(i, 0) + 1
            cv = s.replace(',','').replace('£','').replace('$', '').strip()
            if cv and re.match(r'^-?\d*\.?\d+$', cv):
                num_freq[i] = num_freq.get(i, 0) + 1
                
    col_map = {'date': -1, 'out': -1, 'in': -1, 'bal': -1}
    if date_freq: col_map['date'] = max(date_freq, key=date_freq.get)
    # Filter for numeric columns that appear consistently
    valid_nums = sorted([idx for idx, count in num_freq.items() if count > (len(all_rows) * 0.05)], reverse=True)
    
    if len(valid_nums) >= 1: col_map['bal'] = valid_nums[0]
    if len(valid_nums) >= 2: col_map['in'] = valid_nums[1]
    if len(valid_nums) >= 3: col_map['out'] = valid_nums[2]

    first_amt_col = min([c for c in [col_map['out'], col_map['in'], col_map['bal']] if c != -1] or [99])
    blacklist = ["opening balance", "closing balance", "payments in", "payments out", "payment type and details"]
    
    txns = []
    sticky_date = ""
    
    for row in all_rows:
        vals = [str(v).strip() if v is not None and str(v).lower() != 'nan' else "" for v in row]
        row_text = " ".join(vals).lower()
        
        # Skip summary headers, but keep 'Brought Forward' rows
        is_summary = any(item in row_text for item in blacklist)
        is_forward = "brought forward" in row_text or "carried forward" in row_text
        if is_summary and not is_forward:
            continue
            
        def get_safe(idx):
            return vals[idx] if (idx != -1 and idx < len(vals)) else ""

        # Update date if found
        d_val = get_safe(col_map['date'])
        d_match = re.search(r'\d{1,2}\s[A-Za-z]{3}\s\d{2}', d_val)
        if d_match:
            sticky_date = d_match.group()
        
        row_out = get_safe(col_map['out']).replace(',','').replace('£','').strip()
        row_in = get_safe(col_map['in']).replace(',','').replace('£','').strip()
        row_bal = get_safe(col_map['bal']).replace(',','').replace('£','').strip()
        
        has_amt = any(re.match(r'^-?\d*\.?\d+$', x) for x in [row_out, row_in] if x)
        # Force a new row if we have an amount, a date, or it's a balance forward row
        if d_match or has_amt or is_forward:
            desc_start = col_map['date'] + 1 if col_map['date'] != -1 else 0
            description = " ".join([v for v in vals[desc_start:min(first_amt_col, len(vals))] if v])
            
            # Use sticky date to fill gaps at page turns
            txns.append({
                'Date': sticky_date,
                'Description': description,
                'Paid Out': row_out if row_out and row_out != "0.00" else "",
                'Paid In': row_in if row_in and row_in != "0.00" else "",
                'Balance': row_bal
            })
        elif txns and sticky_date:
            # Check if this is a secondary description line (no date, no amount)
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
            
            # NEW: Collect ALL rows from ALL tables into a master list first
            document_rows = []
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
                        # Convert this specific table to a sorted list of lists
                        sorted_rows = sorted(grid.keys())
                        for r_idx in sorted_rows:
                            row_data = [grid[r_idx].get(c_idx, "") for c_idx in sorted(grid[r_idx].keys())]
                            document_rows.append(row_data)

            try: s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
            except: pass

            if not document_rows:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}
            
            # Process the combined list of rows in one pass
            final_df = process_all_rows(document_rows).drop_duplicates()
            final_df = final_df.astype(str).replace(['nan', 'None', 'NaN', 'inf', '-inf'], '')
            preview_data = final_df.head(100).to_dict(orient="records")

            return {
                "status": "COMPLETED",
                "preview": preview_data,
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Logic Error: {str(e)}", "detail": traceback.format_exc()})

@app.get("/")
def health(): return {"status": "V36 - Global Row Consolidation"}
