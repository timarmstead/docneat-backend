from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import re
import boto3
import os
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_textract_client():
    return boto3.client(
        'textract',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def parse_hsbc_logic(df):
    if df.empty: return pd.DataFrame()
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    txns = []
    current_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        # Skip empty rows
        if not any(vals): continue
        
        d_match = re.search(date_regex, vals[0])
        if d_match:
            current_date = d_match.group()
            
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        p_out = vals[3].replace(',','').replace('£','').strip() if len(vals) > 3 else ""
        p_in = vals[4].replace(',','').replace('£','').strip() if len(vals) > 4 else ""
        bal = vals[5].replace(',','').replace('£','').strip() if len(vals) > 5 else ""

        if current_date and (desc or p_out or p_in):
            has_money = any(c.isdigit() for c in (p_out + p_in))
            if has_money:
                txns.append({
                    'Date': current_date, 'Description': desc,
                    'Paid Out': p_out if p_out else "0",
                    'Paid In': p_in if p_in else "0",
                    'Balance': bal if bal else ""
                })
            elif txns:
                txns[-1]['Description'] += " " + desc
    return pd.DataFrame(txns)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # CRITICAL FIX: Ensure we read raw bytes and don't modify them
        file_bytes = await file.read()
        
        # Verify file isn't empty
        if len(file_bytes) == 0:
            return JSONResponse(status_code=400, content={"error": "File is empty"})

        client = get_textract_client()
        
        # Call Textract with explicit Bytes object
        try:
            response = client.analyze_document(
                Document={'Bytes': file_bytes}, 
                FeatureTypes=['TABLES']
            )
        except client.exceptions.UnsupportedDocumentException:
            return {
                "preview": [{"Date": "ERROR", "Description": "Textract says this PDF format is unsupported. Try 'Save as PDF' on the file first.", "Paid Out": 0, "Paid In": 0, "Balance": 0}],
                "csv_content": "Unsupported Format"
            }
        except Exception as e:
            return {
                "preview": [{"Date": "ERROR", "Description": f"AWS Error: {str(e)}", "Paid Out": 0, "Paid In": 0, "Balance": 0}],
                "csv_content": "AWS Error"
            }

        # Mapping and Parsing
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        all_dfs = []
        
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = txt
            
            df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            all_dfs.append(parse_hsbc_logic(df))

        if not all_dfs:
            return {"preview": [{"Date": "N/A", "Description": "No tables found", "Paid Out": 0, "Paid In": 0, "Balance": 0}], "csv_content": ""}

        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Format for display
        for col in ['Paid Out', 'Paid In', 'Balance']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        preview = final_df.head(50).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        
        stream = io.StringIO()
        final_df.to_csv(stream, index=False)
        
        return {
            "preview": preview,
            "csv_content": stream.getvalue()
        }

    except Exception as e:
        return {"preview": [{"Date": "CRASH", "Description": str(e), "Paid Out": 0, "Paid In": 0, "Balance": 0}], "csv_content": "Error"}

@app.get("/")
def health(): return {"status": "V22 - PDF Byte Handling Fixed"}
