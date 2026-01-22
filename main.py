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

# Enable CORS
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
    """Parse Textract response to extract tables as DataFrames."""
    blocks = response.get('Blocks', [])
    tables = []
    cell_map = {block['Id']: block for block in blocks if block['BlockType'] == 'CELL'}
    table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']

    for table_block in table_blocks:
        table_data = {}
        max_row = 0
        max_col = 0
        
        for rel in table_block.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cell_id in rel['Ids']:
                    cell = cell_map.get(cell_id)
                    if cell:
                        r = cell.get('RowIndex', 0)
                        c = cell.get('ColumnIndex', 0)
                        max_row = max(max_row, r)
                        max_col = max(max_col, c)
                        
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
            row = [table_data.get((r, c), "") for c in range(1, max_col + 1)]
            grid.append(row)
            
        if grid:
            # We don't set headers yet, we clean the raw grid first
            df = pd.DataFrame(grid)
            tables.append(df)
            
    return tables

def clean_dataframe(df):
    """
    Strict filter to remove addresses and bank metadata.
    Only keeps rows that look like bank transactions.
    """
    if df.empty:
        return df

    # 1. Identify rows that start with a Date (e.g., 15 Sep 25)
    # This is the most reliable way to kill the 'Secretary' and 'Address' rows
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    
    # Check first column for date
    is_transaction = df.iloc[:, 0].astype(str).str.contains(date_regex, na=False)
    
    # 2. Filter the dataframe
    df = df[is_transaction].copy()

    if df.empty:
        return df

    # 3. Handle Column Names - Force Competitor Format
    # We drop any extra columns Textract might have hallucinated
    df = df.iloc[:, :5] 
    df.columns = ['Date', 'Description', 'Paid Out', 'Paid In', 'Balance']

    # 4. Clean Numeric Values
    for col in ['Paid Out', 'Paid In', 'Balance']:
        df[col] = df[col].astype(str).str.replace(r'[£,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 5. Drop Noise Phrases
    noise = ['BALANCE BROUGHT FORWARD', 'BALANCE CARRIED FORWARD', 'Account Summary']
    pattern = '|'.join(noise)
    df = df[~df['Description'].astype(str).str.contains(pattern, case=False, na=False)]

    return df.reset_index(drop=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file.filename}"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    final_df = pd.DataFrame()

    try:
        response = textract.analyze_document(
            Document={'Bytes': contents},
            FeatureTypes=['TABLES']
        )
        raw_tables = extract_tables_from_textract(response)
        
        cleaned_list = []
        for table in raw_tables:
            cleaned = clean_dataframe(table)
            if not cleaned.empty:
                cleaned_list.append(cleaned)
        
        if cleaned_list:
            final_df = pd.concat(cleaned_list, ignore_index=True)
            
    except Exception as e:
        print(f"Extraction error: {e}")

    # Fallback to Camelot
    if final_df.empty:
        try:
            tables = camelot.read_pdf(str(input_path), flavor='stream', pages='all')
            cleaned_list = [clean_dataframe(t.df) for t in tables]
            final_df = pd.concat(cleaned_list, ignore_index=True)
        except:
            pass

    # Save ONLY CSV as requested
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    final_df.to_csv(csv_path, index=False)

    preview_data = final_df.head(5).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")

    return {
        "preview": preview_data,
        "csv_url": f"/download/{csv_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    file = OUTPUT_DIR / name
    if not file.exists():
        return {"error": "File not found"}
    return FileResponse(file, media_type="text/csv", filename="docneat-converted.csv")

@app.get("/")
def root():
    return {"message": "DocNeat API - Clean Mode Active"}
