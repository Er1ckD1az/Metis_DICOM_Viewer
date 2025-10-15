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
import numpy as np
from typing import List, Optional
from pathlib import Path

# Import segmentation model
from inference import (
    BraTSSegmentationModel,
    detect_modality_from_filename,
    find_sibling_files,
    load_all_modalities
)

class MRI(BaseModel):
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


class ModalityDetectionResponse(BaseModel):
    mri_id: int
    file_name: str
    detected_modality: str
    sibling_files: dict  # {modality: file_name}


class SegmentationResponse(BaseModel):
    mri_id: int
    segmentation_id: int
    message: str
    segmentation_path: str
    summary: dict  # Statistics about the segmentation


# Global model instance (loaded once)
segmentation_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global segmentation_model

    # Initialize database
    init_database()

    # Load segmentation model from S3
    model_path = os.getenv('MODEL_CHECKPOINT_PATH')
    if model_path:
        print(f"\nLoading segmentation model from: {model_path}")
        try:
            segmentation_model = BraTSSegmentationModel(model_path)
            print("Segmentation model loaded successfully\n")
        except Exception as e:
            print(f"Failed to load segmentation model: {e}")
            print("Segmentation endpoints will not be available\n")

    yield

    # Shutdown: clear data
    print("Application shutting down... clearing all data")
    clear_db_and_cloud()


load_dotenv()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT')
    )


s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

S3_BUCKET = os.getenv('S3_BUCKET')


def init_database():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Main MRI table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS dicom_mri_db
                           (
                               id
                               SERIAL
                               PRIMARY
                               KEY,
                               file_name
                               VARCHAR
                           (
                               255
                           ) NOT NULL,
                               file_path VARCHAR
                           (
                               500
                           ) NOT NULL,
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

            # Segmentation results table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS segmentations
                           (
                               id
                               SERIAL
                               PRIMARY
                               KEY,
                               mri_id
                               INTEGER
                               REFERENCES
                               dicom_mri_db
                           (
                               id
                           ) ON DELETE CASCADE,
                               segmentation_path VARCHAR
                           (
                               500
                           ) NOT NULL,
                               date_created TIMESTAMP NOT NULL,
                               background_voxels INTEGER,
                               necrotic_voxels INTEGER,
                               edema_voxels INTEGER,
                               enhancing_voxels INTEGER
                               );
                           """)

            conn.commit()
            print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()


def get_mri_metadata(file_path):
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


def save_mri_metadata(metadata: dict, file_name: str, file_path: str, file_size: int):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
                       INSERT INTO dicom_mri_db (file_name, file_path, file_size, date_created,
                                                 dim_num, dim_x, dim_y, dim_z, pixdim_x, pixdim_y, pixdim_z)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                       """, (file_name, file_path, file_size, datetime.now(),
                             metadata['dim_num'], metadata['dim_x'], metadata['dim_y'], metadata['dim_z'],
                             metadata['pixdim_x'], metadata['pixdim_y'], metadata['pixdim_z']))
        mri_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
    return mri_id


def upload_to_s3(file_data: bytes, file_name: str):
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=file_name,
        Body=file_data,
        ContentType='application/octet-stream'
    )
    return f"s3://{S3_BUCKET}/{file_name}"


def download_from_s3(file_name: str) -> bytes:
    #Download file from S3 and return as bytes
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_name)
    return response['Body'].read()


def clear_db_and_cloud():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name FROM dicom_mri_db")
            file_names = cursor.fetchall()

            for file_name in file_names:
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=file_name[0])
                except:
                    pass

            # Also delete segmentation files
            cursor.execute("SELECT segmentation_path FROM segmentations")
            seg_paths = cursor.fetchall()
            for seg_path in seg_paths:
                seg_file = seg_path[0].split('/')[-1]
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=seg_file)
                except:
                    pass

            cursor.execute("DELETE FROM segmentations")
            cursor.execute("DELETE FROM dicom_mri_db")
            cursor.execute("ALTER SEQUENCE dicom_mri_db_id_seq RESTART WITH 1")
            cursor.execute("ALTER SEQUENCE segmentations_id_seq RESTART WITH 1")
            conn.commit()
    finally:
        conn.close()

@app.post("/mri", response_model=MRIUploadResponse)
async def upload_mri(file: UploadFile = File(...)):
    #Upload a single MRI file
    if not file.filename.lower().endswith(('.nii', '.nii.gz')):
        raise HTTPException(status_code=400, detail="Only .nii and .nii.gz files are accepted")

    file_data = await file.read()
    file_size = len(file_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
        temp_file.write(file_data)
        temp_file_path = temp_file.name

    try:
        metadata = get_mri_metadata(temp_file_path)
        file_path = upload_to_s3(file_data, file.filename)
        mri_id = save_mri_metadata(metadata, file.filename, file_path, file_size)

        return MRIUploadResponse(
            message="File uploaded successfully",
            mri_id=mri_id,
            file_path=file_path
        )
    finally:
        os.unlink(temp_file_path)


@app.get("/mri", response_model=List[MRIResponse])
async def get_all_mri_metadata():
    #Get all MRI metadata
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM dicom_mri_db")
            results = cursor.fetchall()

            return [
                MRIResponse(
                    id=row[0], file_name=row[1], file_path=row[2], file_size=row[3],
                    date_created=row[4], dim_num=row[5], dim_x=row[6], dim_y=row[7],
                    dim_z=row[8], pixdim_x=row[9], pixdim_y=row[10], pixdim_z=row[11]
                ) for row in results
            ]
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        conn.close()


@app.get("/mri/{mri_id}", response_model=MRIResponse)
async def get_mri_metadata_by_id(mri_id: int):
    #Get MRI metadata by ID
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")

            return MRIResponse(
                id=row[0], file_name=row[1], file_path=row[2], file_size=row[3],
                date_created=row[4], dim_num=row[5], dim_x=row[6], dim_y=row[7],
                dim_z=row[8], pixdim_x=row[9], pixdim_y=row[10], pixdim_z=row[11]
            )
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        conn.close()


@app.get("/mri/{mri_id}/data")
async def get_mri_by_id(mri_id: int):
    #Download MRI file by ID
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name, file_path FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")

            file_name, file_path = row
    finally:
        conn.close()

    try:
        file_content = download_from_s3(file_name)

        return StreamingResponse(
            io.BytesIO(file_content),
            media_type='application/octet-stream',
            headers={"Content-Length": str(len(file_content))}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")


@app.delete("/mri/{mri_id}")
async def delete_mri_by_id(mri_id: int):
    #Delete MRI by ID
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

@app.post("/mri/{mri_id}/detect", response_model=ModalityDetectionResponse)
async def detect_modality(mri_id: int):

    #Detect the modality of the uploaded MRI and find sibling files
    #This is called when user clicks "Detect" button

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")

            file_name = row[0]
    finally:
        conn.close()

    # Detect modality
    detected_modality = detect_modality_from_filename(file_name)

    if not detected_modality:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect modality from filename: {file_name}. "
                   f"Expected format: patientID_modality.nii[.gz] where modality is one of: flair, t1, t1ce, t2"
        )

    # Download the file temporarily to find siblings
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / file_name
        file_data = download_from_s3(file_name)
        temp_path.write_bytes(file_data)

        try:
            sibling_paths = find_sibling_files(str(temp_path), detected_modality)
            sibling_files = {mod: Path(path).name for mod, path in sibling_paths.items()}
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"Missing sibling files: {str(e)}. "
                       f"Please ensure all 4 modalities (flair, t1, t1ce, t2) are uploaded with consistent naming."
            )

    return ModalityDetectionResponse(
        mri_id=mri_id,
        file_name=file_name,
        detected_modality=detected_modality,
        sibling_files=sibling_files
    )

@app.post("/mri/{mri_id}/segment", response_model=SegmentationResponse)
async def segment_mri(mri_id: int):

    #Run segmentation on the MRI
    #This is called after detection, when user wants to predict

    if segmentation_model is None:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded. Check MODEL_CHECKPOINT_PATH environment variable."
        )

    # Get MRI info
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT file_name FROM dicom_mri_db WHERE id = %s", (mri_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"MRI with id {mri_id} not found")

            file_name = row[0]
    finally:
        conn.close()

    # Detect modality and find sibling files
    detected_modality = detect_modality_from_filename(file_name)
    if not detected_modality:
        raise HTTPException(status_code=400, detail="Could not detect modality from filename")

    # Download all 4 modality files from S3
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Download the uploaded file
        file_data = download_from_s3(file_name)
        temp_path = temp_dir_path / file_name
        temp_path.write_bytes(file_data)

        # Find sibling files
        try:
            sibling_paths = find_sibling_files(str(temp_path), detected_modality)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Download all sibling files from S3 to temp directory
        downloaded_paths = {}
        for modality, original_path in sibling_paths.items():
            s3_filename = Path(original_path).name

            # Check if already downloaded (the original uploaded file)
            if s3_filename == file_name:
                downloaded_paths[modality] = str(temp_path)
            else:
                # Download from S3
                try:
                    sibling_data = download_from_s3(s3_filename)
                    local_path = temp_dir_path / s3_filename
                    local_path.write_bytes(sibling_data)
                    downloaded_paths[modality] = str(local_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Could not download {modality} file from S3: {s3_filename}. "
                               f"Please ensure all 4 modalities are uploaded."
                    )

        # Load all volumes into memory
        try:
            volume_dict = load_all_modalities(downloaded_paths)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading volumes: {str(e)}")

        # Run segmentation
        print(f"\n🧠 Running segmentation for MRI ID: {mri_id}")
        try:
            prediction = segmentation_model.predict_volume(volume_dict)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

        # Save segmentation result
        seg_filename = f"seg_{mri_id}_{file_name}"
        seg_temp_path = temp_dir_path / seg_filename

        # Use the first modality file as reference for header/affine
        reference_path = downloaded_paths['flair']
        segmentation_model.save_prediction(prediction, reference_path, str(seg_temp_path))

        # Upload segmentation to S3
        with open(seg_temp_path, 'rb') as f:
            seg_data = f.read()
        seg_s3_path = upload_to_s3(seg_data, seg_filename)

        # Calculate statistics
        unique, counts = np.unique(prediction, return_counts=True)
        class_counts = dict(zip(unique, counts))

        summary = {
            'background_voxels': int(class_counts.get(0, 0)),
            'necrotic_voxels': int(class_counts.get(1, 0)),
            'edema_voxels': int(class_counts.get(2, 0)),
            'enhancing_voxels': int(class_counts.get(3, 0)),
            'total_tumor_voxels': int(class_counts.get(1, 0) + class_counts.get(2, 0) + class_counts.get(3, 0)),
            'shape': prediction.shape
        }

        # Save to database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                               INSERT INTO segmentations (mri_id, segmentation_path, date_created,
                                                          background_voxels, necrotic_voxels, edema_voxels,
                                                          enhancing_voxels)
                               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                               """, (mri_id, seg_s3_path, datetime.now(),
                                     summary['background_voxels'], summary['necrotic_voxels'],
                                     summary['edema_voxels'], summary['enhancing_voxels']))
                seg_id = cursor.fetchone()[0]
                conn.commit()
        finally:
            conn.close()

        return SegmentationResponse(
            mri_id=mri_id,
            segmentation_id=seg_id,
            message="Segmentation completed successfully",
            segmentation_path=seg_s3_path,
            summary=summary
        )

@app.get("/mri/{mri_id}/segmentation")
async def get_segmentation(mri_id: int):
    #Get the latest segmentation for an MRI
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                           SELECT id,
                                  segmentation_path,
                                  date_created,
                                  background_voxels,
                                  necrotic_voxels,
                                  edema_voxels,
                                  enhancing_voxels
                           FROM segmentations
                           WHERE mri_id = %s
                           ORDER BY date_created DESC LIMIT 1
                           """, (mri_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"No segmentation found for MRI ID {mri_id}")

            return {
                'segmentation_id': row[0],
                'segmentation_path': row[1],
                'date_created': row[2],
                'summary': {
                    'background_voxels': row[3],
                    'necrotic_voxels': row[4],
                    'edema_voxels': row[5],
                    'enhancing_voxels': row[6],
                    'total_tumor_voxels': row[4] + row[5] + row[6]
                }
            }
    finally:
        conn.close()

@app.get("/mri/{mri_id}/segmentation/data")
async def download_segmentation(mri_id: int):
    #Download the segmentation mask file
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                           SELECT segmentation_path
                           FROM segmentations
                           WHERE mri_id = %s
                           ORDER BY date_created DESC LIMIT 1
                           """, (mri_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"No segmentation found for MRI ID {mri_id}")

            seg_path = row[0]
            seg_filename = seg_path.split('/')[-1]
    finally:
        conn.close()

    try:
        seg_data = download_from_s3(seg_filename)

        return StreamingResponse(
            io.BytesIO(seg_data),
            media_type='application/octet-stream',
            headers={
                "Content-Disposition": f"attachment; filename={seg_filename}",
                "Content-Length": str(len(seg_data))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving segmentation: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
