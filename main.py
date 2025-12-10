from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tabula
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
    col_map = {
        "Date": "Date", "Transaction Date": "Date", "Posting Date": "Date",
        "Description": "Description", "Particulars": "Description",
        "Amount": "Amount", "Debit": "Amount", "Credit": "Amount"
    }
    df = df.rename(columns=lambda x: col_map.get(x.strip(), x.strip()))
    # Combine amount columns if separate
    if "Amount Out" in df.columns and "Amount In" in df.columns:
        df["Amount"] = df["Amount In"].fillna(0) - df["Amount Out"].fillna(0)
        df = df.drop(["Amount Out", "Amount In"], axis=1)
    # Clean amount
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors='coerce')
    # Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
    # Drop empty rows
    df = df.dropna(how="all")
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

    # Try tabula first
    try:
        dfs = tabula.read_pdf(str(input_path), pages="all", lattice=True, multiple_tables=True)
        if dfs and not all(d.empty for d in dfs):
            df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
            df = clean_dataframe(df)
    except:
        pass

    # Fallback: OCR with pytesseract
    if df.empty:
        images = convert_from_bytes(contents)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        
        # Line-by-line parsing for multi-line entries
        lines = text.split('\n')
        data = []
        current_date = ''
        current_desc = ''
        current_paid_out = ''
        current_paid_in = ''
        current_balance = ''

        for line in lines:
            line = line.strip()
            if not line or "BALANCE BROUGHT FORWARD" in line or "Account Summary" in line or "Opening Balance" in line or "Closing Balance" in line or "Sortcode" in line:
                continue

            # Detect new date
            if re.match(r'\d{1,2} [A-Za-z]{3} 25', line):
                if current_date:
                    amount = float(current_paid_in.replace(',', '') or 0) - float(current_paid_out.replace(',', '') or 0)
                    data.append({"Date": current_date, "Description": current_desc.strip(), "Paid Out": current_paid_out, "Paid In": current_paid_in, "Balance": current_balance, "Amount": amount})
                current_date = line
                current_desc = ''
                current_paid_out = ''
                current_paid_in = ''
                current_balance = ''
            else:
                # Append to description or match numbers
                numbers = re.findall(r'[\d,]+\.?\d*', line)
                if len(numbers) >= 1:
                    current_balance = numbers[-1] if "BALANCE" not in line else ''
                    if len(numbers) >= 3:
                        current_paid_out = numbers[0]
                        current_paid_in = numbers[1]
                    elif len(numbers) == 2:
                        current_paid_out = numbers[0]
                        current_paid_in = numbers[1]
                    elif len(numbers) == 1:
                        if "Visa Rate" in line or "Transaction Fee" in line:
                            current_paid_out = numbers[0]
                else:
                    current_desc += ' ' + line

        # Add the last entry
        if current_date:
            amount = float(current_paid_in.replace(',', '') or 0) - float(current_paid_out.replace(',', '') or 0)
            data.append({"Date": current_date, "Description": current_desc.strip(), "Paid Out": current_paid_out, "Paid In": current_paid_in, "Balance": current_balance, "Amount": amount})

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
    return {"message": "DocNeat Backend Ready!"}
