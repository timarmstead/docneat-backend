from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tabula
import easyocr
import re
import uuid
import shutil
from pathlib import Path

app = FastAPI(title="DocNeat Backend")

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# OCR reader (English)
reader = easyocr.Reader(['en'], gpu=False)

def clean_dataframe(df):
    if df.empty:
        return df
    col_map = {
        "Date": "Date", "Transaction Date": "Date", "Posting Date": "Date",
        "Description": "Description", "Particulars": "Description", "Narration": "Description",
        "Amount": "Amount", "Debit": "Amount", "Credit": "Amount"
    }
    df = df.rename(columns=col_map)
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors='coerce')
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
    return df.dropna(how="all").reset_index(drop=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    # Save file
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Try tabula first (digital PDFs)
    try:
        dfs = tabula.read_pdf(str(input_path), pages="all", lattice=True, multiple_tables=True)
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = clean_dataframe(df)
            if not df.empty:
                excel_path = OUTPUT_DIR / f"{file_id}.xlsx"
                csv_path = OUTPUT_DIR / f"{file_id}.csv"
                df.to_excel(excel_path, index=False)
                df.to_csv(csv_path, index=False)
                input_path.unlink()
                return {
                    "preview": df.head(3).to_dict(orient="records"),
                    "excel_url": f"/download/excel/{excel_path.name}",
                    "csv_url": f"/download/csv/{csv_path.name}"
                }
    except:
        pass

    # Fallback: OCR
    result = reader.readtext(str(input_path))
    text = " ".join([t[1] for t in result])
    pattern = r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(.+?)\s+([-+]?\$?[\d,]+\.?\d*)"
    matches = re.findall(pattern, text)
    data = [{"Date": d, "Description": desc.strip(), "Amount": amt.replace(",", "")} for d, desc, amt in matches]
    df = pd.DataFrame(data)
    df = clean_dataframe(df)

    excel_path = OUTPUT_DIR / f"{file_id}.xlsx"
    csv_path = OUTPUT_DIR / f"{file_id}.csv"
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)
    input_path.unlink()

    return {
        "preview": df.head(3).to_dict(orient="records"),
        "excel_url": f"/download/excel/{excel_path.name}",
        "csv_url": f"/download/csv/{csv_path.name}"
    }

@app.get("/download/excel/{name}")
async def download_excel(name: str):
    file = OUTPUT_DIR / name
    return FileResponse(file, filename="docneat-converted.xlsx")

@app.get("/download/csv/{name}")
async def download_csv(name: str):
    file = OUTPUT_DIR / name
    return FileResponse(file, filename="docneat-converted.csv", media_type="text/csv")

@app.get("/")
def root():
    return {"message": "DocNeat Backend Ready!"}
