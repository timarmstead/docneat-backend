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
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def clean_dataframe(df):
    if df.empty:
        return df
    # Standardize columns
    df.columns = [col.strip() for col in df.columns]
    col_map = {
        "Date": "Date",
        "Payment type and details": "Description",
        "Paid out": "Paid Out",
        "Paid in": "Paid In",
        "Balance": "Balance"
    }
    df = df.rename(columns=col_map)
    # Compute net Amount
    if "Paid Out" in df.columns and "Paid In" in df.columns:
        df['Paid Out'] = pd.to_numeric(df['Paid Out'].str.replace(',', '').str.replace('£', ''), errors='coerce').fillna(0)
        df['Paid In'] = pd.to_numeric(df['Paid In'].str.replace(',', '').str.replace('£', ''), errors='coerce').fillna(0)
        df['Amount'] = df['Paid In'] - df['Paid Out']
    # Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
    # Drop empty or header rows
    df = df.dropna(how="all")
    df = df[~df['Description'].str.contains('BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD|Account Summary|Opening Balance|Closing Balance|Sortcode|Sheet Number|HSBC > UK|Contact tel|Text phone|www.hsbc.co.uk|Y our Statement', na=False, case=False)]
    df = df[df['Amount'] != 0]  # Drop zero amounts
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

    # Try camelot for table extraction (better for complex layouts)
    try:
        tables = camelot.read_pdf(str(input_path), flavor='stream', pages='all')
        if tables and not all(t.df.empty for t in tables):
            df = pd.concat([t.df for t in tables if not t.df.empty], ignore_index=True)
            df = clean_dataframe(df)
    except:
        pass

    # Fallback: OCR with pytesseract
    if df.empty:
        images = convert_from_bytes(contents)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        
        # Improved parser: process line by line
        lines = [line.strip() for line in text.split('\n') if line.strip() and not re.match(r'(The Secretary|Account Name|Your BUSINESS CURRENT ACCOUNT details|Account Summary|Opening Balance|Payments In|Payments Out|Closing Balance|International Bank Account Number|Branch Identifier Code|Sortcode|Sheet Number|46 The Broadway Ealing London W5 5JR|HSBC > UK|Contact tel|Text phone|used by deaf or speech impaired customers|www.hsbc.co.uk)', line)]

        data = []
        current_date = None
        current_desc = ''
        current_paid_out = ''
        current_paid_in = ''
        current_balance = ''

        for line in lines:
            # Detect new date (e.g., "15 Jun 25")
            date_match = re.match(r'\d{1,2} \w{3} 25', line)
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

            # Detect amounts (find numbers with decimals, remove commas)
            amounts = re.findall(r'\d{1,3}(?:,\d{3})*?\.\d{2}', line)
            amounts = [float(a.replace(',', '')) for a in amounts]  # Remove commas and convert to float
            if amounts:
                if len(amounts) == 1:
                    current_balance = amounts[0]
                elif len(amounts) == 2:
                    current_paid_out = amounts[0]
                    current_paid_in = amounts[1]
                elif len(amounts) >= 3:
                    current_paid_out = amounts[0]
                    current_paid_in = amounts[1]
                    current_balance = amounts[-1]
                line = re.sub(r'\d{1,3}(?:,\d{3})*?\.\d{2}', '', line).strip()
            # Append to description
            if line:
                current_desc += ' ' + line if current_desc else line

            # Handle special lines like "Visa Rate" or "DR Non-Sterling"
            if 'Visa Rate' in line or 'DR Non-Sterling' in line:
                # Append to current desc, but look for the amount in the line
                amount_match = re.search(r'\d{1,3}(?:,\d{3})*?\.\d{2}', line)
                if amount_match:
                    amount = float(amount_match.group(0).replace(',', ''))
                    if 'Visa Rate' in line or 'Transaction Fee' in line:
                        current_paid_out = amount

        # Add the last entry
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

    # Save Excel
    excel_path = OUTPUT_DIR / f"{file_id}.xlsx"
    df.to_excel(excel_path, index=False)

    return {
        "preview": df.head(3).to_dict(orient="records"),
        "excel_url": f"/download/{excel_path.name}"
    }

@app.get("/download/{name}")
async def download(name: str):
    file = OUTPUT_DIR / name
    return FileResponse(file, filename="docneat-converted.xlsx")

@app.get("/")
def root():
    return {"message": "DocNeat Backend Ready v5!"}
