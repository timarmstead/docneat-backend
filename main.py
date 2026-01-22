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

# Move Client inside a try-except to catch init errors
def get_client():
    try:
        return boto3.client(
            'textract',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    except Exception:
        return None

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        client = get_client()
        
        if client is None:
            return JSONResponse(status_code=500, content={"error": "Boto3 Client Init Failed"})

        # Try Textract but catch the specific crash
        try:
            response = client.analyze_document(Document={'Bytes': file_bytes}, FeatureTypes=['TABLES'])
        except Exception as aws_err:
            # If AWS fails, we return a clear message instead of a 500 error
            return {
                "preview": [{"Date": "ERROR", "Description": f"AWS Failed: {str(aws_err)}", "Paid Out": 0, "Paid In": 0, "Balance": 0}],
                "csv_content": f"Error,{str(aws_err)}"
            }

        # Parsing Logic
        blocks = response.get('Blocks', [])
        bmap = {b['Id']: b for b in blocks}
        dfs = []
        
        for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
            rows = {}
            for rel in table.get('Relationships', []):
                for cid in rel['Ids']:
                    cell = bmap[cid]
                    r, c = cell['RowIndex'], cell['ColumnIndex']
                    txt = " ".join([bmap[w]['Text'] for r2 in cell.get('Relationships', []) for w in r2['Ids']])
                    rows.setdefault(r, {})[c] = txt
            
            df = pd.DataFrame.from_dict(rows, orient='index').sort_index(axis=1)
            
            # Simple HSBC logic
            date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
            clean_rows = []
            curr_date = None
            for _, row_data in df.iterrows():
                vals = [str(v).strip() for v in row_data.values]
                if re.search(date_regex, vals[0]): curr_date = re.search(date_regex, vals[0]).group()
                if curr_date:
                    clean_rows.append({
                        'Date': curr_date,
                        'Description': vals[1] if len(vals) > 1 else "",
                        'Paid Out': vals[3] if len(vals) > 3 else "0",
                        'Paid In': vals[4] if len(vals) > 4 else "0",
                        'Balance': vals[5] if len(vals) > 5 else "0"
                    })
            dfs.append(pd.DataFrame(clean_rows))

        if not dfs:
            return {"preview": [], "csv_content": ""}

        final_df = pd.concat(dfs, ignore_index=True)
        # JSON Safety
        preview = final_df.head(20).replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        
        stream = io.StringIO()
        final_df.to_csv(stream, index=False)
        
        return {
            "preview": preview,
            "csv_content": stream.getvalue()
        }

    except Exception as e:
        # Final catch-all to prevent the 500 error
        return JSONResponse(status_code=200, content={
            "preview": [{"Date": "CRASH", "Description": str(e), "Paid Out": 0, "Paid In": 0, "Balance": 0}],
            "csv_content": "Internal Error"
        })

@app.get("/")
def health(): return {"status": "V20 Diagnostic Ready"}
