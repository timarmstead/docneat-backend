from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import camelot
import pytesseract
from pdf2image import convert_from_bytes
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

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_tables_from_textract(response: Dict) -> List[pd.DataFrame]:
    """Parse Textract response to extract tables as DataFrames."""
    blocks = response.get('Blocks', [])
    tables = []
    current_table = []
    current_row = []
    cell_map = {block['Id']: block for block in blocks if block['BlockType'] == 'CELL'}
    table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']

    for table_block in table_blocks:
        for rel in table_block.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cell_id in rel['Ids']:
                    cell = cell_map.get(cell_id)
                    if cell:
                        row_index = cell.get('RowIndex', 0)
                        col_index = cell.get('ColumnIndex', 0)
                        
                        if current_row and row_index > current_row[-1]['row']:
                            current_table.append(current_row)
                            current_row = []
                        
                        text = ''
                        for child_rel in cell.get('Relationships', []):
                            if child_rel['Type'] == 'CHILD':
                                for word_id in child_rel['Ids']:
                                    word_block = next((b for b in blocks if b['Id'] == word_id and b['BlockType'] == 'WORD'), None)
                                    if word_block:
                                        text += word_block['Text'] + ' '
                        
                        current_row.append({
                            'row': row_index,
                            'col': col_index,
                            'text': text.strip()
                        })
        
        if current_row:
            current_table.append(current_row)
        
        if current_table:
            max_cols = max(len(row) for row in current_table)
            df_data = []
            for row in current_table:
                sorted_row = sorted(row, key=lambda x: x['col'])
                df_data.append([cell['text'] for cell in sorted_row])
            
            headers = df_data[0] if df_data else []
            data_rows = df_data[1:] if len(df_data) > 1 else []
            df = pd.DataFrame(data_rows, columns=headers)
            tables.append(df)
        
        current_table = []
    
    return tables

def clean_dataframe(df):
    if df.empty:
        return df
    df.columns = [col.strip() for col in df.columns]
    col_map = {
        "Date": "Date",
        "Payment type and details": "Description",
        "Paid out": "Paid Out",
        "Paid in": "Paid In",
        "Balance": "Balance"
    }
    df = df.rename(columns=col_map)
    if "Paid Out" in df.columns and "Paid In" in df.columns:
        df['Paid Out'] = pd.to_numeric(df['Paid Out'].str.replace(',', '').str.replace('£', ''), errors='coerce').fillna(0)
        df['Paid In'] = pd.to_numeric(df['Paid In'].str.replace(',', '').str.replace('£', ''), errors='coerce').fillna(0)
        df['Amount'] = df['Paid In'] - df['Paid Out']
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
    df = df.dropna(how="all")
    df = df[~df.get('Description', pd.Series()).str.contains(
        'BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD|Account Summary|Opening Balance|Closing Balance|Sortcode|Sheet Number|HSBC > UK|Contact tel|Text phone|www.hsbc.co.uk|Your Statement',
        na=False, case=False, regex=True)]
    df = df[df['Amount'] != 0]
    df = df.reset_index(drop=True)
    return df

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file.filename}"
    
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    df = pd.DataFrame()

    # Primary: AWS Textract
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

    # Fallback 1: Camelot
    if df.empty:
        try:
            tables = camelot.read_pdf(str(input_path), flavor='stream', pages='all')
            if tables and not all(t.df.empty for t in tables):
                df = pd.concat([t.df for t in tables if not t.df.empty], ignore_index=True)
                df = clean_dataframe(df)
        except:
            pass

    # Fallback 2: OCR with pytesseract
    if df.empty:
        images = convert_from_bytes(contents)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        
        lines = [line.strip() for line in text.split('\n') 
                 if line.strip() and not re.match(
                     r'(The Secretary|Account Name|Your BUSINESS CURRENT ACCOUNT details|Account Summary|Opening Balance|Payments In|Payments Out|Closing Balance|International Bank Account Number|Branch Identifier Code|Sortcode|Sheet Number|46 The Broadway Ealing London W5 5JR|HSBC > UK|Contact tel|Text phone|www.hsbc.co.uk)',
                     line)]

        data = []
        current_date = None
        current_desc = ''
        current_paid_out = ''
        current_paid_in = ''
        current_balance = ''

        for line in lines:
            date_match = re.match(r'\d{1,2} \w{3} \d{2}', line)
            if date_match:
                if current_date:
                    amount = float(current_paid_in or 0) - float(current_paid_out or 0)
                    data.append({
                        'Date': current_date,
                        'Description': current_desc.strip(),
                        'Paid Out': current_paid_out,
                        'Paid In': current_paid_in,
                        'Balance': current_balance,
                        'Amount': amount
                    })
                current_date = date_match.group(0)
                current_desc = line.replace(current_date, '').strip()
                current_paid_out = ''
                current_paid_in = ''
                current_balance = ''
                continue

            amounts = re.findall(r'\d{1,3}(?:,\d{3})*?\.\d{2}', line)
            amounts = [float(a.replace(',', '')) for a in amounts]
            if amounts:
                if len(amounts) == 1:
                    if 'Visa Rate' in line or 'Transaction Fee' in line:
                        current_paid_out = amounts[0]
                    else:
                        current_balance = amounts[0]
                elif len(amounts) == 2:
                    current_paid_out = amounts[0]
                    current_paid_in = amounts[1]
                elif len(amounts) >= 3:
                    current_paid_out = amounts[0]
                    current_paid_in = amounts[1]
                    current_balance = amounts[2]
                line = re.sub(r'\d{1,3}(?:,\d{3})*?\.\d{2}', '', line).strip()
            if line:
                current_desc += ' ' + line if current_desc else line

        if current_date:
            amount = float(current_paid_in or 0) - float(current_paid_out or 0)
            data.append({
                'Date': current_date,
                'Description': current_desc.strip(),
                'Paid Out': current_paid_out,
                'Paid In': current_paid_in,
                'Balance': current_balance,
                'Amount': amount
            })

        df = pd.DataFrame(data)
        df = clean_dataframe(df)

    # Save both Excel and CSV
    excel_path = OUTPUT_DIR / f"{file_id}.xlsx"
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)

    # Safe preview: replace NaN/inf/-inf with None before JSON serialization
    preview_data = (
        df.head(3)
          .replace([np.nan, np.inf, -np.inf], None)
          .to_dict(orient="records")
    )

    return {
        "preview": preview_data,
        "excel_url": f"/download/{excel_path.name}",
        "csv_url": f"/download/{csv_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    file = OUTPUT_DIR / name
    if file.suffix == '.csv':
        media_type = "text/csv"
        filename = "docneat-converted.csv"
    else:
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "docneat-converted.xlsx"
    
    return FileResponse(file, media_type=media_type, filename=filename)

@app.get("/")
def root():
    return {"message": "DocNeat Backend Ready v8 - Textract + CSV support"}
