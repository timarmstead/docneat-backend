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

# Enable CORS for frontend integration
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
                        
                        # Extract text from words within the cell
                        text = ''
                        for child_rel in cell.get('Relationships', []):
                            if child_rel['Type'] == 'CHILD':
                                for word_id in child_rel['Ids']:
                                    word_block = next((b for b in blocks if b['Id'] == word_id), None)
                                    if word_block and 'Text' in word_block:
                                        text += word_block['Text'] + ' '
                        
                        table_data[(r, c)] = text.strip()

        # Build DataFrame from grid
        grid = []
        for r in range(1, max_row + 1):
            row = [table_data.get((r, c), "") for c in range(1, max_col + 1)]
            grid.append(row)
            
        if grid:
            df = pd.DataFrame(grid[1:], columns=grid[0])
            tables.append(df)
            
    return tables

def clean_dataframe(df):
    """
    Strict cleaning logic to match competitor output.
    Removes headers, footers, and non-transaction rows.
    """
    if df.empty:
        return df

    # Standardize column names
    df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
    
    # Map messy headers to standard names (matches bankstatementconverter sample)
    col_map = {
        "Date": "Date",
        "Payment type and details": "Description",
        "Pay m e nt t y pe and de t ails": "Description", # Handle Textract artifacts
        "Paid out": "Paid Out",
        "Paid in": "Paid In",
        "Balance": "Balance"
    }
    df = df.rename(columns=col_map)

    # 1. Date Anchor Filter: Every transaction MUST start with a date (e.g., '15 Sep 25')
    # This automatically deletes addresses, IBANs, and footers.
    date_pattern = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    df = df[df['Date'].astype(str).str.contains(date_pattern, na=False)]

    # 2. Description Noise Filter: Remove internal bank markers
    noise_phrases = [
        'Account Summary', 'Opening Balance', 'Closing Balance', 
        'Sortcode', 'Sheet Number', 'www.hsbc.co.uk', 'Your Statement'
    ]
    pattern = '|'.join(noise_phrases)
    df = df[~df['Description'].astype(str).str.contains(pattern, case=False, na=False)]

    # 3. Numeric Formatting: Remove currency symbols and commas
    for col in ['Paid Out', 'Paid In', 'Balance']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[£,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Final calculation for net Amount
    if 'Paid Out' in df.columns and 'Paid In' in df.columns:
        df['Amount'] = df['Paid In'] - df['Paid Out']

    return df.reset_index(drop=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file.filename}"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    df = pd.DataFrame()

    # Step 1: AWS Textract (Primary Extraction)
    try:
        response = textract.analyze_document(
            Document={'Bytes': contents},
            FeatureTypes=['TABLES']
        )
        tables = extract_tables_from_textract(response)
        if tables:
            df = pd.concat(tables, ignore_index=True)
            df = clean_dataframe(df)
    except Exception as e:
        print(f"Textract error: {e}")

    # Step 2: Fallback to Camelot if Textract fails or returns empty
    if df.empty:
        try:
            tables = camelot.read_pdf(str(input_path), flavor='stream', pages='all')
            if tables:
                df = pd.concat([t.df for t in tables], ignore_index=True)
                df = clean_dataframe(df)
        except:
            pass

    # Save outputs
    excel_path = OUTPUT_DIR / f"{file_id}.xlsx"
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)

    preview_data = df.head(5).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")

    return {
        "preview": preview_data,
        "excel_url": f"/download/{excel_path.name}",
        "csv_url": f"/download/{csv_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    file = OUTPUT_DIR / name
    media_type = "text/csv" if file.suffix == '.csv' else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    filename = f"docneat-converted{file.suffix}"
    return FileResponse(file, media_type=media_type, filename=filename)

@app.get("/")
def root():
    return {"message": "DocNeat Backend Refactored - Clean Extraction Active"}
