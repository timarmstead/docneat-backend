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

# Strict Date Regex: Matches "16 Sep 25"
DATE_REGEX = r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2}'

def is_clean_num(val):
    """Checks if a string is EXCLUSIVELY a number (no words allowed)."""
    s = str(val).replace('£','').replace('$','').replace(',','').strip()
    return bool(re.fullmatch(r'-?\d+\.\d{2}', s))

def to_num(val):
    s = str(val).replace('£','').replace('$','').replace(',','').strip()
    return s if re.fullmatch(r'-?\d+\.\d{2}', s) else ""

def parse_hsbc_rows(df, sticky_date):
    txns = []
    num_cols = len(df.columns)
    
    # Identify which columns are generally used for amounts (usually the last 3)
    # We don't hardcode them, but we use them as a guide.
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        row_str = " ".join(vals).lower()
        
        # 1. Skip Headers and Summary Noise
        if any(x in row_str for x in ["opening balance", "closing balance", "payments in", "payments out", "payment type"]):
            if not "forward" in row_str: continue

        # 2. Extract Date
        d_match = re.search(DATE_REGEX, vals[0] + " " + (vals[1] if num_cols > 1 else ""))
        if d_match: sticky_date = d_match.group()

        # 3. Extract Amounts and Balance
        # Logic: Balance is always the last clean number. 
        # Paid In is the one before it. Paid Out is the one before that.
        amounts = []
        for i, v in enumerate(vals):
            if is_clean_num(v):
                amounts.append({'idx': i, 'val': to_num(v)})
        
        p_out, p_in, p_bal = "", "", ""
        used_indices = []

        if "forward" in row_str:
            # For Forward rows, we only care about the balance (usually the last number)
            if amounts:
                p_bal = amounts[-1]['val']
                used_indices = [amounts[-1]['idx']]
        else:
            # Standard row: Identify Out/In/Bal based on right-to-left order
            if len(amounts) >= 1:
                p_bal = amounts[-1]['val']
                used_indices.append(amounts[-1]['idx'])
            if len(amounts) >= 2:
                # If there are two more numbers, it's Out and In. 
                # But HSBC rows usually only have ONE transaction amount + Balance.
                # If we find 3 numbers total: Out, In, Balance.
                if len(amounts) == 3:
                    p_in = amounts[-2]['val']
                    p_out = amounts[-3]['val']
                    used_indices.extend([amounts[-2]['idx'], amounts[-3]['idx']])
                else:
                    # Only 2 numbers: Is it an amount and a balance?
                    # If the column index is far to the right, it's likely Paid In.
                    # Otherwise, Paid Out.
                    if amounts[-2]['idx'] >= (num_cols - 2):
                        p_in = amounts[-2]['val']
                    else:
                        p_out = amounts[-2]['val']
                    used_indices.append(amounts[-2]['idx'])

        # 4. Description: Everything that isn't a Date or a used Amount
        desc_parts = []
        for i, v in enumerate(vals):
            # If the cell has text or was NOT used as a clean number, it's description
            if i not in used_indices and v:
                # Don't include the date string again in description
                if not re.search(DATE_REGEX, v):
                    desc_parts.append(v)
        desc = " ".join(desc_parts).strip()

        # 5. Save the row
        if d_match or p_out or p_in or p_bal:
            txns.append({
                'Date': sticky_date,
                'Description': desc,
                'Paid Out': p_out,
                'Paid In': p_in,
                'Balance': p_bal
            })
        elif txns and desc and not any(k in desc.lower() for k in ["page", "details"]):
            # Append orphaned text
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
                        table_txns, sticky_date = parse_hsbc_rows(df_table, sticky_date)
                        final_data.extend(table_txns)

            try: s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
            except: pass

            if not final_data:
                return {"status": "COMPLETED", "preview": [], "csv_content": ""}
            
            final_df = pd.DataFrame(final_data).drop_duplicates()
            final_df = final_df.astype(str).replace(['nan', 'None', 'NaN', '0.00'], '')
            
            return {
                "status": "COMPLETED",
                "preview": final_df.head(100).to_dict(orient="records"),
                "csv_content": final_df.to_csv(index=False)
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Logic Error: {str(e)}", "detail": traceback.format_exc()})

@app.get("/")
def health(): return {"status": "V41 - Visual Anchor Logic Active"}
