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

# IMPROVED REGEX: Matches "16 Sep 25" OR "05 Oct"
DATE_REGEX = r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s\d{2,4})?'

def is_clean_num(val):
    if not val: return False
    s = str(val).replace('£','').replace('$','').replace(',','').strip()
    return bool(re.fullmatch(r'-?\d+\.\d{2}', s))

def to_num(val):
    s = str(val).replace('£','').replace('$','').replace(',','').strip()
    return s if re.fullmatch(r'-?\d+\.\d{2}', s) else ""

def find_columns_semantically(df):
    col_map = {'date': -1, 'out': -1, 'in': -1, 'bal': -1}
    keywords = {
        'date': ["date", "trans", "value", "when"],
        'out': ["paid out", "debit", "withdrawals", "spent", "money out", "out"],
        'in': ["paid in", "credit", "deposits", "received", "money in", "in"],
        'bal': ["balance", "running balance", "total", "bal"]
    }

    for r_idx in range(min(3, len(df))):
        row = [str(c).lower().strip() for c in df.iloc[r_idx].values]
        for c_idx, cell_text in enumerate(row):
            for key, key_list in keywords.items():
                if any(k in cell_text for k in key_list) and col_map[key] == -1:
                    col_map[key] = c_idx

    # Fallback to visual anchors
    num_cols = len(df.columns)
    col_stats = {i: {'nums': 0, 'dates': 0} for i in range(num_cols)}
    for _, row in df.iterrows():
        for i, v in enumerate(row.values):
            s = str(v).strip()
            if re.search(DATE_REGEX, s): col_stats[i]['dates'] += 1
            if is_clean_num(s): col_stats[i]['nums'] += 1

    if col_map['date'] == -1:
        col_map['date'] = max(col_stats, key=lambda k: col_stats[k]['dates']) if any(c['dates'] > 0 for c in col_stats.values()) else 0
    
    num_indices = [i for i, c in col_stats.items() if c['nums'] > 0]
    if col_map['bal'] == -1 and len(num_indices) >= 1: col_map['bal'] = num_indices[-1]
    if col_map['in'] == -1 and len(num_indices) >= 2: col_map['in'] = num_indices[-2]
    if col_map['out'] == -1 and len(num_indices) >= 3: col_map['out'] = num_indices[-3]

    return col_map

def parse_bank_agnostic(df, sticky_date, global_year):
    if df.empty: return [], sticky_date
    
    col_map = find_columns_semantically(df)
    txns = []
    blacklist = ["opening balance", "closing balance", "payments in", "payments out", "payment type", "fscs"]

    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        row_str = " ".join(vals).lower()
        
        if any(x in row_str for x in blacklist) and not "forward" in row_str:
            continue

        d_match = re.search(DATE_REGEX, vals[col_map['date']]) if col_map['date'] != -1 else None
        if d_match: 
            sticky_date = d_match.group()
            # If no year in the date, add the global year found at top of statement
            if global_year and len(sticky_date.split()) < 3:
                sticky_date = f"{sticky_date} {global_year}"

        p_out = to_num(vals[col_map['out']]) if col_map['out'] != -1 else ""
        p_in = to_num(vals[col_map['in']]) if col_map['in'] != -1 else ""
        p_bal = to_num(vals[col_map['bal']]) if col_map['bal'] != -1 else ""
        
        used_indices = {col_map['date'], col_map['out'], col_map['in'], col_map['bal']}
        desc = " ".join([v for i, v in enumerate(vals) if i not in used_indices and v and not re.search(DATE_REGEX, v)]).strip()

        if "forward" in row_str or "start balance" in row_str:
            actual_bal = p_bal or p_in or p_out
            txns.append({'Date': sticky_date, 'Description': desc, 'Paid Out': '', 'Paid In': '', 'Balance': actual_bal})
        elif d_match or p_out or p_in or p_bal:
            txns.append({'Date': sticky_date, 'Description': desc, 'Paid Out': p_out, 'Paid In': p_in, 'Balance': p_bal})
        elif txns and desc and not any(k in desc.lower() for k in ["page", "details"]):
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
            
            # Find the Year globally at the top of the statement
            global_year = ""
            for block in all_blocks:
                if block['BlockType'] == 'LINE':
                    text = block.get('Text', '')
                    # Look for 4-digit years like 2021 or 2024
                    year_match = re.search(r'\b(20\d{2})\b', text)
                    if year_match:
                        global_year = year_match.group(1)
                        break

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
                        table_txns, sticky_date = parse_bank_agnostic(df_table, sticky_date, global_year)
                        final_data.extend(table_txns)

            try: s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
            except: pass

            if not final_data:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}
            
            columns = ['Date', 'Description', 'Paid Out', 'Paid In', 'Balance']
            final_df = pd.DataFrame(final_data, columns=columns).drop_duplicates()
            final_df = final_df.astype(str).replace(['nan', 'None', 'NaN', '0.00'], '')
            
            return {
                "status": "COMPLETED",
                "preview": final_df.head(100).to_dict(orient="records"),
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Logic Error: {str(e)}", "detail": traceback.format_exc()})

@app.get("/")
def health(): return {"status": "V45 - Year Grafting & Flex Date Active"}
