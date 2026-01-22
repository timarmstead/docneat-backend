from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import re
import uuid
from pathlib import Path
import boto3
import os
import numpy as np
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

textract = boto3.client(
    'textract',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def extract_tables_optimized(response: Dict) -> List[pd.DataFrame]:
    blocks = response.get('Blocks', [])
    block_map = {b['Id']: b for b in blocks}
    tables = []
    for table_block in [b for b in blocks if b['BlockType'] == 'TABLE']:
        table_data = {}
        max_row, max_col = 0, 0
        for rel in table_block.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cell_id in rel['Ids']:
                    cell = block_map.get(cell_id)
                    if not cell: continue
                    r, c = cell.get('RowIndex', 0), cell.get('ColumnIndex', 0)
                    max_row, max_col = max(max_row, r), max(max_col, c)
                    text = " ".join([block_map[w]['Text'] for rel_child in cell.get('Relationships', []) if rel_child['Type'] == 'CHILD' for w in rel_child['Ids'] if w in block_map])
                    table_data[(r, c)] = text.strip()
        grid = [[table_data.get((r, c), "") for c in range(1, max_col + 1)] for r in range(1, max_row + 1)]
        if grid: tables.append(pd.DataFrame(grid))
    return tables

def clean_bank_data(df):
    if df.empty: return pd.DataFrame()
    
    # 1. Map columns (Date is usually 0, Money is usually the end)
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    data_rows = []
    last_date = None
    
    # We find the header row to know where money is
    money_cols = []
    for c in range(df.shape[1]):
        col_str = " ".join(df.iloc[:, c].astype(str)).replace(',', '')
        if pd.to_numeric(col_str.replace('£','').split(), errors='coerce').notnull().any():
            money_cols.append(c)
    
    if not money_cols: return pd.DataFrame()
    
    out_idx = money_cols[0]
    in_idx = money_cols[1] if len(money_cols) > 1 else money_cols[0]
    bal_idx = money_cols[-1]

    for i in range(len(df)):
        row = df.iloc[i].astype(str).tolist()
        row_text = " ".join(row)
        
        # Skip noise
        if any(x in row_text for x in ["Account Summary", "Sheet Number", "HBUKGB"]): continue
        
        # Check for date
        date_match = re.search(date_regex, row[0])
        if date_match:
            last_date = date_match.group()
        
        # A valid row must either have a date OR have a description with money
        desc = row[1] if len(row) > 1 else ""
        if len(row) > 2 and not desc.strip(): # Sometimes Textract shifts desc to col 2
            desc = row[2]

        paid_out = row[out_idx].replace(',', '').replace('£', '').strip()
        paid_in = row[in_idx].replace(',', '').replace('£', '').strip()
        balance = row[bal_idx].replace(',', '').replace('£', '').strip()

        # Only add if there is a description or money involved
        if last_date and (desc.strip() or paid_out or paid_in):
            data_rows.append({
                'Date': last_date,
                'Description': desc.strip(),
                'Paid Out': paid_out,
                'Paid In': paid_in,
                'Balance': balance
            })

    # Final Cleanup: Merge rows that are just description continuations
    final_rows = []
    for r in data_rows:
        # If this row has no money and no date change, it's a continuation of the previous description
        if not r['Paid Out'] and not r['Paid In'] and final_rows and r['Date'] == final_rows[-1]['Date']:
            final_rows[-1]['Description'] += " " + r['Description']
        else:
            final_rows.append(r)

    res_df = pd.DataFrame(final_rows)
    for col in ['Paid Out', 'Paid In', 'Balance']:
        if col in res_df.columns:
            res_df[col] = pd.to_numeric(res_df[col], errors='coerce').fillna(0.0)
    
    # Remove rows that are just balance carries
    res_df = res_df[~res_df['Description'].str.contains('BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD', case=False, na=False)]
    return res_df

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file.filename}"
    contents = await file.read()
    with open(input_path, "wb") as f: f.write(contents)

    all_dfs = []
    response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
    tables = extract_tables_optimized(response)
    
    for t in tables:
        cleaned = clean_bank_data(t)
        if not cleaned.empty:
            all_dfs.append(cleaned)

    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=['Date', 'Description', 'Paid Out', 'Paid In', 'Balance'])
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    final_df.to_csv(csv_path, index=False)

    return {
        "preview": final_df.head(10).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records"),
        "csv_url": f"/download/{csv_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    return FileResponse(OUTPUT_DIR / name, media_type="text/csv", filename="docneat-export.csv")

@app.get("/")
def root(): return {"status": "DocNeat Multi-Line Engine Active"}
