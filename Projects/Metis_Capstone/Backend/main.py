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
from gradio_client import Client, handle_file
from inference import detect_modality_from_filename


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
    sibling_files: dict


class SegmentationResponse(BaseModel):
    mri_id: int
    segmentation_id: int
    message: str
    segmentation_path: str
    summary: dict


#Global Gradio client
gradio_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gradio_client

    init_database()

    #Initialize Gradio client for HF Space
    hf_space = os.getenv('HF_SPACE_URL', 'EdTheProgrammer/Metis-DICOM-Backend')
    print(f"\n Initializing Gradio client for HF Space: {hf_space}")
    try:
        gradio_client = Client(hf_space)
        print(" Gradio client initialized successfully")
        print("Segmentation will be handled by HuggingFace Space\n")
    except Exception as e:
        print(f"  Failed to initialize Gradio client: {e}")
        print("Segmentation endpoints may not work properly\n")

    yield

    #On shutdown clear data
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
            #Main MRI table
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

            #Segmentation results table with model_type
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
                               enhancing_voxels INTEGER,
                               model_type VARCHAR
                           (
                               50
                           ) DEFAULT 'pspnet'
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

    detected_modality = detect_modality_from_filename(file_name)

    if not detected_modality:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect modality from filename: {file_name}. "
                   f"Expected format: patientID_modality.nii[.gz] where modality is one of: flair, t1, t1ce, t2"
        )

    patient_id = file_name.rsplit('_', 1)[0]
    modalities = ['flair', 't1', 't1ce', 't2']
    sibling_files = {}
    missing = []

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for modality in modalities:
                for ext in ['.nii.gz', '.nii']:
                    expected_filename = f"{patient_id}_{modality}{ext}"
                    cursor.execute("SELECT file_name FROM dicom_mri_db WHERE file_name = %s", (expected_filename,))
                    result = cursor.fetchone()

                    if result:
                        sibling_files[modality] = result[0]
                        break

                if modality not in sibling_files:
                    missing.append(f"{patient_id}_{modality}.nii[.gz]")
    finally:
        conn.close()

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Missing required modality files: {', '.join(missing)}. "
                   f"Please ensure all 4 modalities (flair, t1, t1ce, t2) are uploaded with consistent naming."
        )

    return ModalityDetectionResponse(
        mri_id=mri_id,
        file_name=file_name,
        detected_modality=detected_modality,
        sibling_files=sibling_files
    )


@app.post("/mri/{mri_id}/segment", response_model=SegmentationResponse)
async def segment_mri(mri_id: int, model_type: str = "pspnet"):

    if gradio_client is None:
        raise HTTPException(
            status_code=503,
            detail="Gradio client not initialized. Check HF_SPACE_URL environment variable."
        )

    print(f"\n Starting segmentation for MRI ID: {mri_id}")
    print(f"   Model type: {model_type}")

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

    #Find all 4 modality files if present, otherwise copy file for all
    patient_id = file_name.rsplit('_', 1)[0]
    modalities = ['flair', 't1', 't1ce', 't2']
    s3_filenames = {}

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for modality in modalities:
                for ext in ['.nii.gz', '.nii']:
                    expected_filename = f"{patient_id}_{modality}{ext}"
                    cursor.execute(
                        "SELECT file_name FROM dicom_mri_db WHERE file_name = %s",
                        (expected_filename,)
                    )
                    result = cursor.fetchone()

                    if result:
                        s3_filenames[modality] = result[0]
                        break
    finally:
        conn.close()

    if len(s3_filenames) != 4:
        raise HTTPException(
            status_code=404,
            detail=f"Missing modality files. Found: {list(s3_filenames.keys())}"
        )

    print(f" Found all modalities: {list(s3_filenames.keys())}")

    #Download all 4 modality files from S3 and save to temp directory
    print(" Downloading files from S3...")
    temp_dir = Path(tempfile.mkdtemp())
    temp_files = {}

    try:
        for modality, s3_filename in s3_filenames.items():
            try:
                file_data = download_from_s3(s3_filename)
                print(f"    {modality}: {len(file_data) / 1024 / 1024:.1f} MB")

                # Save to temp file
                temp_file_path = temp_dir / f"{modality}.nii"
                temp_file_path.write_bytes(file_data)
                temp_files[modality] = str(temp_file_path)

            except Exception as e:
                # Clean up on error
                for temp_file in temp_files.values():
                    try:
                        Path(temp_file).unlink()
                    except:
                        pass
                try:
                    temp_dir.rmdir()
                except:
                    pass
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download {modality}: {str(e)}"
                )

        #Call HuggingFace Space using Gradio Client
        print(f"\n Calling HF Space via Gradio Client")
        print(f"   Using {'Fusion' if model_type == 'fusion' else 'PSPNet'} model...")

        try:
            #Call the Gradio endpoint
            result = gradio_client.predict(
                input_file=handle_file(temp_files['t1']),  # Your HF Space expects single file
                api_name="/segment_brain_tumor"
            )

            print(" Received response from HF Space")
            print(f"   Result type: {type(result)}")

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid response format from HF Space: {type(result)}"
                )

            seg_file_path = result[0]  #Path to segmentation file
            summary_text = result[1]  #Summary statistics text

            print(f"   Segmentation file: {seg_file_path}")
            print(f"   Summary preview: {summary_text[:100] if summary_text else 'N/A'}...")

        except Exception as e:
            error_msg = f"Failed to call HF Space: {str(e)}"
            print(f" {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # Read segmentation file
        print("Reading segmentation mask...")
        try:
            seg_file_path_obj = Path(seg_file_path)

            if not seg_file_path_obj.exists():
                raise FileNotFoundError(f"Segmentation file not found: {seg_file_path}")

            seg_bytes = seg_file_path_obj.read_bytes()
            print(f"    Read file: {len(seg_bytes) / 1024 / 1024:.1f} MB")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read segmentation file: {str(e)}"
            )

        #Save segmentation to S3
        print(" Uploading to S3...")
        seg_filename = f"seg_{mri_id}_{file_name}"
        seg_s3_path = upload_to_s3(seg_bytes, seg_filename)
        print(f"    Saved to: {seg_s3_path}")

        #Parse summary text to extract statistics
        print(" Parsing statistics...")
        import re
        stats = {}

        #Parse lines like "- Background: 1,234,567 voxels"
        if summary_text:
            matches = re.findall(r'-\s*([^:]+):\s*([\d,]+)\s*voxels', summary_text)

            for name, count in matches:
                clean_count = int(count.replace(',', ''))
                name_lower = name.strip().lower()

                if 'background' in name_lower:
                    stats['background_voxels'] = clean_count
                elif 'necrotic' in name_lower:
                    stats['necrotic_voxels'] = clean_count
                elif 'edema' in name_lower:
                    stats['edema_voxels'] = clean_count
                elif 'enhancing' in name_lower:
                    stats['enhancing_voxels'] = clean_count
                elif 'total' in name_lower:
                    stats['total_tumor_voxels'] = clean_count

        #Create summary dict with defaults
        summary = {
            'background_voxels': stats.get('background_voxels', 0),
            'necrotic_voxels': stats.get('necrotic_voxels', 0),
            'edema_voxels': stats.get('edema_voxels', 0),
            'enhancing_voxels': stats.get('enhancing_voxels', 0),
            'total_tumor_voxels': stats.get('total_tumor_voxels',
                                            stats.get('necrotic_voxels', 0) +
                                            stats.get('edema_voxels', 0) +
                                            stats.get('enhancing_voxels', 0))
        }

        print(f"   Total tumor voxels: {summary['total_tumor_voxels']:,}")

        #Save to database
        print(" Saving to database...")
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                               INSERT INTO segmentations (mri_id, segmentation_path, date_created,
                                                          background_voxels, necrotic_voxels, edema_voxels,
                                                          enhancing_voxels, model_type)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                               """, (mri_id, seg_s3_path, datetime.now(),
                                     summary['background_voxels'], summary['necrotic_voxels'],
                                     summary['edema_voxels'], summary['enhancing_voxels'], model_type))
                seg_id = cursor.fetchone()[0]
                conn.commit()
                print(f"    Segmentation ID: {seg_id}")
        finally:
            conn.close()

        print(f"\n Segmentation complete!\n")

        return SegmentationResponse(
            mri_id=mri_id,
            segmentation_id=seg_id,
            message="Segmentation completed successfully",
            segmentation_path=seg_s3_path,
            summary=summary
        )

    finally:
        #Clean up temp files
        print(" Cleaning up temporary files...")
        for temp_file in temp_files.values():
            try:
                Path(temp_file).unlink()
            except Exception as e:
                print(f"   Warning: Failed to delete {temp_file}: {e}")

        try:
            temp_dir.rmdir()
        except Exception as e:
            print(f"   Warning: Failed to delete temp directory: {e}")


@app.get("/mri/{mri_id}/segmentation")
async def get_segmentation(mri_id: int):
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
                                  enhancing_voxels,
                                  model_type
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
                'file_name': row[1].split('/')[-1],
                'date_created': row[2],
                'model_type': row[7] if len(row) > 7 else 'pspnet',
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