from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tabula
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import re
import uuid
import shutil
from pathlib import Path
import io

app = FastAPI()

# Fix CORS properly
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

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Railway has it pre-installed

def clean_dataframe(df):
    if df.empty:
        return df
    col_map = {
        "Date": "Date", "Transaction Date": "Date", "Posting Date": "Date",
        "Description": "Description", "Particulars": "Description",
        "Amount": "Amount", "Debit": "Amount", "Credit": "Amount"
    }
    df = df.rename(columns=lambda x: col_map.get(x.strip(), x.strip()))
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors='coerce')
    return df.dropna(how="all").reset_index(drop=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
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
        
        pattern = r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(.+?)\s+([-+]?\$?[\d,]+\.?\d*)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        data = [{"Date": d.strip(), "Description": desc.strip(), "Amount": amt.replace(",", "").replace("$", "")} for d, desc, amt in matches]
        df = pd.DataFrame(data)
        df = clean_dataframe(df)

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
