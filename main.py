from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import camelot
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

# AWS Textract client setup
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
textract = boto3.client(
    'textract',
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def extract_tables_from_textract(response: Dict) -> List[pd.DataFrame]:
    blocks = response.get('Blocks', [])
    tables = []
    cell_map = {block['Id']: block for block in blocks if block['BlockType'] == 'CELL'}
    table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']

    for table_block in table_blocks:
        table_data = {}
        max_row, max_col = 0, 0
        for rel in table_block.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cell_id in rel['Ids']:
                    cell = cell_map.get(cell_id)
                    if cell:
                        r, c = cell.get('RowIndex', 0), cell.get('ColumnIndex', 0)
                        max_row, max_col = max(max_row, r), max(max_col, c)
                        text = ''
                        for child_rel in cell.get('Relationships', []):
                            if child_rel['Type'] == 'CHILD':
                                for word_id in child_rel['Ids']:
                                    word_block = next((b for b in blocks if b['Id'] == word_id), None)
                                    if word_block and 'Text' in word_block:
                                        text += word_block['Text'] + ' '
                        table_data[(r, c)] = text.strip()
        grid = []
        for r in range(1, max_row + 1):
            grid.append([table_data.get((r, c), "") for c in range(1, max_col + 1)])
        if grid:
            tables.append(pd.DataFrame(grid))
    return tables

def clean_dataframe(df):
    if df.empty: return pd.DataFrame()
    
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    col_types = {}
    
    # 1. Dynamically find the Date Column
    for c in range(df.shape[1]):
        if df.iloc[:, c].astype(str).str.contains(date_regex, na=False).any():
            col_types['Date'] = c
            break
    if 'Date' not in col_types: return pd.DataFrame()

    # 2. Dynamically find Money Columns (Out, In, Balance)
    money_cols = []
    for c in range(col_types['Date'] + 1, df.shape[1]):
        sample = df.iloc[:, c].astype(str).str.replace(r'[£,\s]', '', regex=True)
        if pd.to_numeric(sample, errors='coerce').notnull().sum() > 0:
            money_cols.append(c)
    
    if len(money_cols) >= 3:
        col_types['Balance'], col_types['Paid In'], col_types['Paid Out'] = money_cols[-1], money_cols[-2], money_cols[-3]
    elif len(money_cols) == 2:
        col_types['Paid In'], col_types['Paid Out'] = money_cols[-1], money_cols[-2]
    elif len(money_cols) == 1:
        col_types['Paid Out'] = money_cols[0]

    # Description is everything between Date and first Money column
    m_start = min(money_cols) if money_cols else df.shape[1]
    col_types['Desc_Idx'] = list(range(col_types['Date'] + 1, m_start))

    # 3. Process Rows & Merge Multi-line Descriptions
    transactions = []
    curr = None
    first_date_row = df.iloc[:, col_types['Date']].astype(str).str.contains(date_regex, na=False).idxmax()
    
    for i in range(first_date_row, len(df)):
        row = df.iloc[i]
        date_match = re.search(date_regex, str(row.iloc[col_types['Date']]))
        
        if date_match:
            if curr: transactions.append(curr)
            desc = " ".join([str(row.iloc[idx]) for idx in col_types['Desc_Idx'] if str(row.iloc[idx]).lower() != 'nan']).strip()
            curr = {
                'Date': date_match.group(),
                'Description': desc,
                'Paid Out': str(row.iloc[col_types.get('Paid Out', -1)]) if 'Paid Out' in col_types else "0",
                'Paid In': str(row.iloc[col_types.get('Paid In', -1)]) if 'Paid In' in col_types else "0",
                'Balance': str(row.iloc[col_types.get('Balance', -1)]) if 'Balance' in col_types else "0"
            }
        elif curr: # Continuation line
            desc = " ".join([str(row.iloc[idx]) for idx in col_types['Desc_Idx'] if str(row.iloc[idx]).lower() != 'nan']).strip()
            if desc: curr['Description'] += " " + desc
            for k in ['Paid Out', 'Paid In', 'Balance']:
                if k in col_types:
                    val = str(row.iloc[col_types[k]]).strip()
                    if val and val.lower() != 'nan' and curr[k] in ['0', '0.0', '']:
                        curr[k] = val

    if curr: transactions.append(curr)
    
    # 4. Final Formatting
    res = pd.DataFrame(transactions)
    if not res.empty:
        for col in ['Paid Out', 'Paid In', 'Balance']:
            res[col] = pd.to_numeric(res[col].astype(str).str.replace(r'[£,]', '', regex=True), errors='coerce').fillna(0)
        # Nuke standard bank noise
        res = res[~res['Description'].str.contains('BALANCE BROUGHT FORWARD|Account Summary', case=False, na=False)]
        return res[['Date', 'Description', 'Paid Out', 'Paid In', 'Balance']]
    return pd.DataFrame()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file.filename}"
    contents = await file.read()
    with open(input_path, "wb") as f: f.write(contents)

    cleaned_list = []
    try:
        response = textract.analyze_document(Document={'Bytes': contents}, FeatureTypes=['TABLES'])
        for table in extract_tables_from_textract(response):
            cl = clean_dataframe(table)
            if not cl.empty: cleaned_list.append(cl)
    except Exception as e: print(f"Error: {e}")

    final_df = pd.concat(cleaned_list, ignore_index=True) if cleaned_list else pd.DataFrame()
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    final_df.to_csv(csv_path, index=False)

    return {
        "preview": final_df.head(10).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records"),
        "csv_url": f"/download/{csv_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    return FileResponse(OUTPUT_DIR / name, media_type="text/csv", filename="docneat-transactions.csv")

@app.get("/")
def root(): return {"status": "DocNeat Logic Hardened"}
