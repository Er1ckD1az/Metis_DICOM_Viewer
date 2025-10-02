import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import psycopg2
import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import nibabel as nib
import tempfile
import io
from typing import List

class MRI(BaseModel): # Defining the table structure for postgres db
    id: int
    file_name: str
    file_path: str
    file_size: int # File size in bytes
    date_created: datetime
    dim_num: int # Number of dimensions
    dim_x: int # Width in pixels
    dim_y: int # Height in pixels
    dim_z: int # Number of slices
    pixdim_x: float # Size of pixels (mm)
    pixdim_y: float # Size of pixels (mm)
    pixdim_z: float # Size of slices (mm)

class MRIResponse(BaseModel):
    id: int
    file_name: str
    file_path: str
    file_size: int
    date_created: datetime
    dim_num: int
    dim_x: int
    dim_y: int
    dim_z: int
    pixdim_x: float
    pixdim_y: float
    pixdim_z: float 

class MRIUploadResponse(BaseModel):
    message: str
    mri_id: int
    file_path: str

@asynccontextmanager
async def lifespan(app: FastAPI): # Lifespan manager creates db on startup and clears all data on shutdown
    init_database()
    yield

    print("Application shutting down... clearing all data")
    clear_db_and_cloud()

load_dotenv()
app = FastAPI(lifespan=lifespan)

app.add_middleware( # Middleware allows the frontend to connect to backend
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection(): # Function that connects to the postgres db
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT')
    )

s3_client = boto3.client( # Connecting to the AWS S3 cloud
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

S3_BUCKET = os.getenv('S3_BUCKET')

def init_database(): # Creates database table on startup
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dicom_mri_db (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    file_size INTEGER NOT NULL,
                    date_created TIMESTAMP NOT NULL,
                    dim_num INTEGER NOT NULL,
                    dim_x INTEGER NOT NULL,
                    dim_y INTEGER NOT NULL,
                    dim_z INTEGER NOT NULL,
                    pixdim_x FLOAT NOT NULL,
                    pixdim_y FLOAT NOT NULL,
                    pixdim_z FLOAT NOT NULL
                );
            """)
            conn.commit()
            print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def get_mri_metadata(file_path): # Uses file to return a dictionary with MRI metadata
    img = nib.load(file_path)
    header = img.header
        
    dims = img.shape
    dim_num = len(dims)
    pix_dims = header.get_zooms()
        
    return {
        'dim_num': dim_num,
        'dim_x': dims[0],
        'dim_y': dims[1],
        'dim_z': dims[2],
        'pixdim_x': float(pix_dims[0]),
        'pixdim_y': float(pix_dims[1]),
        'pixdim_z': float(pix_dims[2]),
    }

def save_mri_metadata(metadata: dict, file_name: str, file_path: str, file_size: int): # Inserts file metadata into postgres db
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(f"""
        INSERT INTO dicom_mri_db (file_name, file_path, file_size, date_created, 
                              dim_num, dim_x, dim_y, dim_z, pixdim_x, pixdim_y, pixdim_z)
        VALUES ('{file_name}', '{file_path}', {file_size}, '{datetime.now()}',
                {metadata['dim_num']}, {metadata['dim_x']}, {metadata['dim_y']}, {metadata['dim_z']},
                {metadata['pixdim_x']}, {metadata['pixdim_y']}, {metadata['pixdim_z']})
        RETURNING id;""")
        mri_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
    return mri_id
    
def upload_mri_to_cloud(file_data: bytes, file_name: str): # Uploads the MRI file to the S3 cloud
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=file_name,
        Body=file_data,
        ContentType='application/octet-stream'
    )
    return f"s3://{S3_BUCKET}/{file_name}"

def clear_db_and_cloud(): # Deletes all metadata and files from Postgres and S3
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name FROM dicom_mri_db")
            file_names = cursor.fetchall()
            
            for file_name in file_names:
                s3_client.delete_object(Bucket=S3_BUCKET, Key=file_name[0])
                
            cursor.execute("DELETE FROM dicom_mri_db")
            cursor.execute("ALTER SEQUENCE dicom_mri_db_id_seq RESTART WITH 1")
            conn.commit()
    finally:
        conn.close()

@app.post("/mri", response_model=MRIUploadResponse)
async def upload_mri(file: UploadFile = File(...)): # Saves MRI metadata and uploads file to cloud
    if not file.filename.lower().endswith(('.nii', '.nii.gz')):
        raise HTTPException(status_code=400, detail="Only .nii and .nii.gz files are accepted")
    
    file_data = await file.read()
    file_size = len(file_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
        temp_file.write(file_data)
        temp_file_path = temp_file.name
    
    try:
        metadata = get_mri_metadata(temp_file_path)
        file_path = upload_mri_to_cloud(file_data, file.filename)
        mri_id = save_mri_metadata(metadata, file.filename, file_path, file_size)
        
        return MRIUploadResponse(
            message="File uploaded successfully",
            mri_id=mri_id,
            file_path=file_path
        )
    
    finally:
        os.unlink(temp_file_path)

@app.get("/mri", response_model=List[MRIResponse])
async def get_all_mri_metadata(): # Gets all MRI metadata currently in database
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM dicom_mri_db")
            results = cursor.fetchall()
            
            return [
                MRIResponse(
                    id=row[0],
                    file_name=row[1],
                    file_path=row[2],
                    file_size=row[3],
                    date_created=row[4],
                    dim_num=row[5],
                    dim_x=row[6],
                    dim_y=row[7],
                    dim_z=row[8],
                    pixdim_x=row[9],
                    pixdim_y=row[10],
                    pixdim_z=row[11]
                ) for row in results
            ]
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        conn.close()

@app.get("/mri/{mri_id}", response_model=MRIResponse)
async def get_mri_metadata_by_id(mri_id: int): # Gets MRI metadata by id
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")
            
            return MRIResponse(
                id=row[0],
                file_name=row[1],
                file_path=row[2],
                file_size=row[3],
                date_created=row[4],
                dim_num=row[5],
                dim_x=row[6],
                dim_y=row[7],
                dim_z=row[8],
                pixdim_x=row[9],
                pixdim_y=row[10],
                pixdim_z=row[11]
            )
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        conn.close()

@app.get("/mri/{mri_id}/data")
async def get_mri_by_id(mri_id: int): # Gets the MRI file contents by id
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name, file_path FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")
            
            file_name, file_path = row
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        conn.close()

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_name)
        file_content = response['Body'].read()
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type='application/octet-stream',
            headers={
                "Content-Length": str(len(file_content))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")

@app.delete("/mri/{mri_id}")
async def delete_mri_by_id(mri_id: int): # Deletes metadata and file from database and cloud by id
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")

            s3_client.delete_object(Bucket=S3_BUCKET, Key=row[0])

            cursor.execute("DELETE FROM dicom_mri_db WHERE id = %s", (mri_id,))
            conn.commit()

            return {"message": f"MRI with id {mri_id} deleted successfully."}
    finally:
        conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)