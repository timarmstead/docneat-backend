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

# Strict Date Regex: Matches "16 Sep 25" but NOT "22 EUR 19"
DATE_REGEX = r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2}'

def get_clean_amt(val):
    if not val: return ""
    s = str(val).replace('£','').replace('$','').replace(',','').strip()
    match = re.search(r'-?\d+\.\d{2}', s)
    return match.group(0) if match else ""

def parse_hsbc_logic(df, sticky_date):
    if df.empty: return [], sticky_date
    
    # 1. High-Certainty Column Mapping
    # Date is always column 0 or 1. Amounts are always in the last 3 columns.
    num_cols = len(df.columns)
    col_map = {'date': -1, 'out': -1, 'in': -1, 'bal': -1}
    
    # Scan first 2 columns for date
    for i in range(min(2, num_cols)):
        if df.iloc[:, i].astype(str).str.contains(DATE_REGEX, regex=True).any():
            col_map['date'] = i
            break
            
    # Map from the right: Balance is usually last, then In, then Out
    # We look for numeric patterns in the last 4 columns
    potential_nums = []
    for i in range(max(0, num_cols-4), num_cols):
        if df.iloc[:, i].astype(str).apply(get_clean_amt).replace('', np.nan).notna().any():
            potential_nums.append(i)
    
    if len(potential_nums) >= 1: col_map['bal'] = potential_nums[-1]
    if len(potential_nums) >= 2: col_map['in'] = potential_nums[-2]
    if len(potential_nums) >= 3: col_map['out'] = potential_nums[-3]

    txns = []
    blacklist = ["opening balance", "closing balance", "payments in", "payments out", 
                 "payment type", "credit interest", "debit interest", "fscs"]

    for _, row in df.iterrows():
        vals = [str(v).strip() if v is not None and str(v).lower() != 'nan' else "" for v in row.values]
        row_str = " ".join(vals).lower()
        
        # Immediate removal of summary and footer noise
        if any(item in row_str for item in blacklist) and not "forward" in row_str:
            continue
            
        # Date Logic
        d_idx = col_map['date']
        d_val = vals[d_idx] if d_idx != -1 else ""
        d_match = re.search(DATE_REGEX, d_val)
        if d_match: sticky_date = d_match.group()
        
        # Amount Logic
        p_out = get_clean_amt(vals[col_map['out']]) if col_map['out'] != -1 else ""
        p_in = get_clean_amt(vals[col_map['in']]) if col_map['in'] != -1 else ""
        p_bal = get_clean_amt(vals[col_map['bal']]) if col_map['bal'] != -1 else ""
        
        # Description: Everything that isn't the date or an amount
        money_indices = [col_map['out'], col_map['in'], col_map['bal']]
        desc_parts = []
        for i, v in enumerate(vals):
            if i != col_map['date'] and i not in money_indices and v:
                desc_parts.append(v)
        desc = " ".join(desc_parts)

        # New row trigger: Date found OR any Money found
        has_money = p_out or p_in or p_bal
        if d_match or has_money:
            # Handle Summary Rows
            if "forward" in desc.lower():
                # For Brought/Carried Forward, we only want the Balance column
                final_bal = p_bal or p_in or p_out
                txns.append({'Date': sticky_date, 'Description': desc, 'Paid Out': '', 'Paid In': '', 'Balance': final_bal})
            else:
                txns.append({'Date': sticky_date, 'Description': desc, 'Paid Out': p_out, 'Paid In': p_in, 'Balance': p_bal})
        elif txns and desc:
            # Append multi-line text to previous description
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
            sticky_date = ""
            
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
                        table_txns, sticky_date = parse_hsbc_logic(df_table, sticky_date)
                        final_data.extend(table_txns)

            try: s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
            except: pass

            if not final_data:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}
            
            final_df = pd.DataFrame(final_data).drop_duplicates()
            # Absolute cleanup
            final_df = final_df.astype(str).replace(['nan', 'None', 'NaN', '0.00'], '')
            
            return {
                "status": "COMPLETED",
                "preview": final_df.head(100).to_dict(orient="records"),
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Logic Error: {str(e)}", "detail": traceback.format_exc()})

@app.get("/")
def health(): return {"status": "V40 - Fixed Boundary Constraints Active"}
