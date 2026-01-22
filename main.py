import time
import io
import os
import re
import uuid
import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Standard CORS setup to ensure your frontend can communicate with Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS Clients using Render Environment Variables
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

textract = boto3.client('textract',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

BUCKET_NAME = os.getenv('AWS_S3_BUCKET')

def parse_hsbc_logic(df):
    """
    Parses table data specifically for HSBC layouts.
    Handles 'Non-Sterling Transaction Fees' appearing on secondary lines.
    """
    if df.empty:
        return pd.DataFrame()
    
    date_regex = r'\d{1,2}\s[A-Za-z]{3}\s\d{2}'
    transactions = []
    current_date = None
    
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values]
        
        # Check for date in the first column
        d_match = re.search(date_regex, vals[0])
        if d_match:
            current_date = d_match.group()
            
        # Description is usually in Column 1 or 2
        desc = vals[1] if vals[1].lower() != 'nan' and vals[1] != "" else (vals[2] if len(vals) > 2 else "")
        
        # Money columns: Paid Out(3), Paid In(4), Balance(5)
        p_out = vals[3].replace(',','').replace('£','').strip() if len(vals) > 3 else ""
        p_in = vals[4].replace(',','').replace('£','').strip() if len(vals) > 4 else ""
        bal = vals[5].replace(',','').replace('£','').strip() if len(vals) > 5 else ""

        if current_date and (desc or p_out or p_in):
            # If the row contains numeric values, it's a new transaction line
            if any(c.isdigit() for c in (p_out + p_in)):
                transactions.append({
                    'Date': current_date,
                    'Description': desc,
                    'Paid Out': p_out if p_out else "0",
                    'Paid In': p_in if p_in else "0",
                    'Balance': bal if bal else ""
                })
            # If no money, it's a description continuation (like 'Non-Sterling Fee')
            elif transactions:
                transactions[-1]['Description'] += " " + desc

    return pd.DataFrame(transactions)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Create a unique temporary key for the file in S3
    file_key = f"temp_uploads/{uuid.uuid4()}-{file.filename}"
    
    try:
        # 1. Read file bytes and upload to S3 transit bucket
        content = await file.read()
        s3.put_object(Bucket=BUCKET_NAME, Key=file_key, Body=content)
        
        # 2. Start Asynchronous Analysis (Required for Multi-Page PDFs)
        start_response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': BUCKET_NAME, 'Name': file_key}},
            FeatureTypes=['TABLES']
        )
        job_id = start_response['JobId']
        
        # 3. Wait for AWS to finish processing (Polling)
        status = "IN_PROGRESS"
        while status == "IN_PROGRESS":
            time.sleep(2)  # Pause for 2 seconds to avoid hitting rate limits
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                # Handle pagination if the document has many tables across many pages
                pages = [response]
                next_token = response.get('NextToken')
                while next_token:
                    next_response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                    pages.append(next_response)
                    next_token = next_response.get('NextToken')
                
                # 4. Map the AI blocks back into readable DataFrames
                all_dfs = []
                for page in pages:
                    blocks = page.get('Blocks', [])
                    bmap = {b['Id']: b for b in blocks}
                    
                    for table in [b for b in blocks if b['BlockType'] == 'TABLE']:
                        grid = {}
                        for rel in table.get('Relationships', []):
                            for cid in rel['Ids']:
                                cell = bmap[cid]
                                r, c = cell['RowIndex'], cell['ColumnIndex']
                                # Combine all words in a single cell
                                cell_text = " ".join([
                                    bmap[w]['Text'] for r2 in cell.get('Relationships', []) 
                                    for w in r2['Ids'] if bmap[w]['BlockType'] == 'WORD'
                                ])
                                grid.setdefault(r, {})[c] = cell_text
                        
                        df_raw = pd.DataFrame.from_dict(grid, orient='index').sort_index(axis=1)
                        all_dfs.append(parse_hsbc_logic(df_raw))

                if not all_dfs:
                    return {"preview": [], "csv_content": "", "message": "No tables detected."}

                # Combine all tables from all pages into one master list
                final_df = pd.concat(all_dfs, ignore_index=True)
                
                # Cleanup: ensure numbers are floats and empty strings are 0.0
                for col in ['Paid Out', 'Paid In', 'Balance']:
                    final_df[col] = pd.to_numeric(final_df[col].replace('', '0'), errors='coerce').fillna(0.0)

                # Prepare the data for the frontend
                preview = final_df.head(100).to_dict(orient="records")
                stream = io.StringIO()
                final_df.to_csv(stream, index=False)
                
                return {
                    "preview": preview,
                    "csv_content": stream.getvalue()
                }
            
            if status == 'FAILED':
                return JSONResponse(status_code=500, content={"error": "AWS Textract Job Failed"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        
    finally:
        # 5. NO DATA STORAGE PROMISE: Delete the file from S3 immediately
        try:
            s3.delete_object(Bucket=BUCKET_NAME, Key=file_key)
        except:
            pass # Fail silently if file was already deleted

@app.get("/")
def health_check():
    """Confirms the backend is alive and has the S3 bucket configured."""
    return {
        "status": "DocNeat Industrial V25 Active",
        "s3_configured": bool(BUCKET_NAME)
    }
