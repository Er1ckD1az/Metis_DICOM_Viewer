import React, { useEffect, useRef, useState, useMemo } from "react";
import { useLocation } from "react-router-dom";
import "./DicomViewer.css";
import * as nifti from "nifti-reader-js";
import { Niivue } from "@niivue/niivue";

const API_BASE_URL = 'http://localhost:8000'; // Backend URL

const uploadNiftiFile = async (file: File): Promise<{ 
  mri_id: number; 
  file_path: string; 
  message: string 
}> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/mri`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Upload failed:', response.status, errorText);
    throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
  }

  const result = await response.json();
  console.log('🟢 Upload successful:', result);
  return result;
};

const DicomViewer: React.FC = () => {
  const location = useLocation();
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // Multi-grid refs for 4-viewer grid
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null]);
  const overlayRefs = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null]);
  
  // NiiVue instances for 3D rendering (one per window)
  const nv3DRefs = useRef<Map<number, any>>(new Map());
  
  // Track if demo has been loaded to prevent infinite loop
  const demoLoadedRef = useRef(false);

  const [hasImage, setHasImage] = useState(false);
  const [niftiData, setNiftiData] = useState<Float32Array | null>(null);
  const [dimensions, setDimensions] = useState<[number, number, number]>([0, 0, 0]);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [mriId, setMriId] = useState<number | null>(null);

  const [predictionMask, setPredictionMask] = useState<Float32Array | null>(null);
  const [predictionDimensions, setPredictionDimensions] = useState<[number, number, number]>([0, 0, 0]);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predictionProgress, setPredictionProgress] = useState<string>('');
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [segmentationSummary, setSegmentationSummary] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<'pspnet' | 'unet'>('pspnet');

  // Dynamic windows state with per-window settings
  type ViewType = 'axial' | 'coronal' | 'sagittal' | '3d';
  type WindowState = {
    id: number;
    view: ViewType;
    slice: number;
    windowLevel: number;
    windowWidth: number;
    zoomLevel: number;
    panOffset: { x: number; y: number };
  };
  
  const [dynamicWindows, setDynamicWindows] = useState<WindowState[]>([
    { 
      id: 1, 
      view: 'axial',
      slice: 0,
      windowLevel: 200,
      windowWidth: 600,
      zoomLevel: 1,
      panOffset: { x: 0, y: 0 }
    }
  ]);
  const [selectedWindowId, setSelectedWindowId] = useState<number>(1); // Track selected window
  const nextWindowIdDynamic = useRef(2);
  const isPanning = useRef(false);
  const lastPanPos = useRef({ x: 0, y: 0 });

  // overlay tools state
  const [mode, setMode] = useState<'none'|'measure'|'annotate'|'erase'>('none');
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const drawingRef = useRef<{ drawing: boolean; x:number; y:number } | null>(null);
  const [measurements, setMeasurements] = useState<Array<{ x1:number,y1:number,x2:number,y2:number,windowId:number }>>([]);
  const [annotations, setAnnotations] = useState<Array<{ x:number,y:number,text:string,color:string,windowId:number }>>([]);
  const [hoveredAnnotation, setHoveredAnnotation] = useState<number | null>(null);
  const [cursorPosition, setCursorPosition] = useState<{ x: number; y: number } | null>(null);

  // Annotation modal state
  const [showAnnotationModal, setShowAnnotationModal] = useState(false);
  const [annotationText, setAnnotationText] = useState('');
  const [pendingAnnotation, setPendingAnnotation] = useState<{ x: number; y: number; color: string } | null>(null);

  const [viewerWindows, setViewerWindows] = useState<Array<{
    id: string;
    position: { x: number; y: number };
    size: { width: number; height: number };
    file: File | null;
    nv: any;
    canvasRef: React.RefObject<HTMLCanvasElement | null>;
    title: string;
    isMinimized: boolean;
  }>>([]);


  // Accordion state for collapsible sections
  const [expandedSections, setExpandedSections] = useState<{[key: string]: boolean}>({
    view: false,
    tools: false,
    windows: false,
    transform: false,
    upload: false,
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  // Interaction mode state (only one can be active at a time)
  const [interactionMode, setInteractionMode] = useState<'scroll' | 'zoom' | 'pan' | 'brightness' | 'contrast' | null>(null);

  // Helper functions to ensure only one tool is active at a time across all sections
  const activateInteractionMode = (newMode: 'scroll' | 'zoom' | 'pan' | 'brightness' | 'contrast' | null) => {
    setMode('none'); // Deactivate any measuring tools
    setInteractionMode(newMode);
  };

  const activateMeasuringMode = (newMode: 'none' | 'measure' | 'annotate' | 'erase') => {
    setInteractionMode(null); // Deactivate any interaction tools
    setMode(newMode);
  };

  // Dynamic window management functions
  const addWindow = () => {
    if (dynamicWindows.length >= 4) {
      alert('Maximum of 4 windows allowed');
      return;
    }
    
    // Clear all measurements and annotations when adding a new window
    setMeasurements([]);
    setAnnotations([]);
    setHoveredAnnotation(null);
    
    const newWindow: WindowState = {
      id: nextWindowIdDynamic.current++,
      view: 'axial',
      slice: Math.floor(dimensions[2] / 2),
      windowLevel: 200,
      windowWidth: 600,
      zoomLevel: 1,
      panOffset: { x: 0, y: 0 },
    };
    setDynamicWindows(prev => [...prev, newWindow]);
  };

  const removeWindow = (id: number) => {
    if (dynamicWindows.length === 1) {
      alert('Must have at least one window');
      return;
    }
    
    // Clear all measurements and annotations when removing a window
    setMeasurements([]);
    setAnnotations([]);
    setHoveredAnnotation(null);
    
    setDynamicWindows(prev => prev.filter(w => w.id !== id));
    // If we're removing the selected window, select the first remaining one
    if (id === selectedWindowId) {
      const remaining = dynamicWindows.filter(w => w.id !== id);
      if (remaining.length > 0) {
        setSelectedWindowId(remaining[0].id);
      }
    }
  };

  const changeWindowView = (id: number, view: ViewType) => {
    setDynamicWindows(prev => prev.map(w => {
      if (w.id !== id) return w;
      
      // Set slice to middle of the view
      let middleSlice = 0;
      if (view === 'axial') {
        middleSlice = Math.floor(dimensions[2] / 2);
      } else if (view === 'coronal') {
        middleSlice = Math.floor(dimensions[1] / 2);
      } else if (view === 'sagittal') {
        middleSlice = Math.floor(dimensions[0] / 2);
      }
      
      return { ...w, view, slice: middleSlice };
    }));
  };

  // Update per-window state
  const updateWindowState = (id: number, updates: Partial<WindowState>) => {
    setDynamicWindows(prev => prev.map(w =>
      w.id === id ? { ...w, ...updates } : w
    ));
  };

  // Load and display NIfTI file (same logic as original)
  const loadNiftiFile = async (file: File, skipBackendUpload = false) => {
    try {
      console.log("🟢 loadNiftiFile called with:", file.name);
      setCurrentFile(file); // Save file for 3D rendering

      // Clear previous overlay/segmentation data when loading new file
      setPredictionMask(null);
      setPredictionDimensions([0, 0, 0]);
      setShowOverlay(false);
      setSegmentationSummary(null);

      const arrayBuffer = await file.arrayBuffer();
      console.log("🟢 ArrayBuffer loaded, size:", arrayBuffer.byteLength);
      
      if (!nifti.isNIFTI(arrayBuffer)) {
        console.error("❌ Not a valid NIfTI file");
        throw new Error("Not a valid NIfTI file");
      }
      console.log("🟢 Valid NIfTI file confirmed");
      
      const header = nifti.readHeader(arrayBuffer);
      const dims = [header.dims[1], header.dims[2], header.dims[3]];
      console.log("🟢 Dimensions:", dims);
      
      const dataBuffer = nifti.readImage(header, arrayBuffer);

      // convert to Float32Array depending on datatype (best-effort)
      let data;
      const datatype = (header as any).datatype || 4;
      if (datatype === 4) {
        const int16Data = new Int16Array(dataBuffer);
        data = new Float32Array(int16Data);
      } else if (datatype === 8) {
        const int32Data = new Int32Array(dataBuffer);
        data = new Float32Array(int32Data);
      } else if (datatype === 16) {
        data = new Float32Array(dataBuffer);
      } else {
        const int16Data = new Int16Array(dataBuffer);
        data = new Float32Array(int16Data);
      }

      // Apply same rotation/flip as original to improve orientation
      const [x,y,z] = dims;
      const rotated = new Float32Array(data.length);
      for (let k=0;k<z;k++){
        for (let j=0;j<y;j++){
          for (let i=0;i<x;i++){
            const originalIndex = i + j * x + k * x * y;
            const newI = j;
            const newJ = x - 1 - i;
            const newK = k;
            const flippedJ = y - 1 - newJ;
            const newIndex = newI + flippedJ * y + newK * y * x;
            rotated[newIndex] = data[originalIndex];
          }
        }
      }

      setNiftiData(rotated);
      setDimensions([y, x, z]);
      setHasImage(true);
      
      // Initialize all windows with correct default slice values
      setDynamicWindows(prev => prev.map(window => ({
        ...window,
        slice: window.view === 'axial' ? Math.floor(z / 2) :
               window.view === 'coronal' ? Math.floor(x / 2) :
               window.view === 'sagittal' ? Math.floor(y / 2) :
               0
      })));
      
      console.log("🟢 NIfTI loaded successfully!", {
        dimensions: [y, x, z],
        dataLength: rotated.length,
        hasImage: true
      });

      // Only upload to backend if not skipped (e.g., for demo files that already exist in S3)
      if (!skipBackendUpload) {
        try {
          console.log("Uploading file to backend...");
          const uploadResult = await uploadNiftiFile(file);
          console.log("🟢 File uploaded to backend:", uploadResult);
          setMriId(uploadResult.mri_id);

        } catch (uploadError) {
          console.error("❌ Failed to upload file to backend:", uploadError);
        }
      } else {
        console.log("⏩ Skipping backend upload (file already in S3)");
      }

    } catch (error) {
      console.error("❌ Failed to load NIfTI file:", error);
      alert("Failed to load NIfTI file. Please check the file format.");
    }
  };

  const getSlice = (data: Float32Array, dims: [number,number,number], sliceIndex:number, view:string) => {
    const [x,y,z] = dims;
    if (view === 'axial'){
      const slice = new Float32Array(x*y);
      for (let j=0;j<y;j++){
        for (let i=0;i<x;i++){
          const dataIndex = i + j * x + sliceIndex * x * y;
            slice[i + j * x] = data[dataIndex];
        }
      }
      return { slice, width: x, height: y };
    } else if (view === 'coronal'){
      const slice = new Float32Array(x*z);
      for (let k=0;k<z;k++){
        for (let i=0;i<x;i++){
          const dataIndex = i + sliceIndex * x + k * x * y;
            const flippedK = z - 1 - k;
            slice[i + flippedK * x] = data[dataIndex];
        }
      }
      return { slice, width: x, height: z };
    } else {
      const slice = new Float32Array(y*z);
      for (let k=0;k<z;k++){
        for (let j=0;j<y;j++){
          const dataIndex = sliceIndex + j * x + k * x * y;
            const flippedK = z - 1 - k;
            slice[j + flippedK * y] = data[dataIndex];
        }
      }
      return { slice, width: y, height: z };
    }
  };

  const getOverlaySlice = (
    maskData: Float32Array,
    dims: [number, number, number],
    sliceIndex: number,
    view: string
  ) => {
    const [x, y, z] = dims;
    
    if (view === 'axial') {
      const slice = new Float32Array(x * y);
      for (let j = 0; j < y; j++) {
        for (let i = 0; i < x; i++) {
          const dataIndex = i + j * x + sliceIndex * x * y;
          slice[i + j * x] = maskData[dataIndex];
        }
      }
      return { slice, width: x, height: y };
    } else if (view === 'coronal') {
      const slice = new Float32Array(x * z);
      for (let k = 0; k < z; k++) {
        for (let i = 0; i < x; i++) {
          const dataIndex = i + sliceIndex * x + k * x * y;
          const flippedK = z - 1 - k;
          slice[i + flippedK * x] = maskData[dataIndex];
        }
      }
      return { slice, width: x, height: z };
    } else {
      // sagittal
      const slice = new Float32Array(y * z);
      for (let k = 0; k < z; k++) {
        for (let j = 0; j < y; j++) {
          const dataIndex = sliceIndex + j * x + k * x * y;
          const flippedK = z - 1 - k;
          slice[j + flippedK * y] = maskData[dataIndex];
        }
      }
      return { slice, width: y, height: z };
    }
  };

  const detectModality = async (mriId: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/mri/${mriId}/detect`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to detect modality');
      }

      const result = await response.json();
      console.log('🟢 Modality detection result:', result);
      return result;
    } catch (error) {
      console.error('❌ Modality detection failed:', error);
      throw error;
    }
  };

  const runSegmentation = async (mriId: number, modelType: 'pspnet' | 'unet' = 'pspnet') => {
    try {
      const response = await fetch(`${API_BASE_URL}/mri/${mriId}/segment?model_type=${modelType}`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Segmentation failed');
      }

      const result = await response.json();
      console.log('🟢 Segmentation result:', result);
      return result;
    } catch (error) {
      console.error('❌ Segmentation failed:', error);
      throw error;
    }
  };

  const downloadSegmentationMask = async (mriId: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/mri/${mriId}/segmentation/data`);

      if (!response.ok) {
        throw new Error('Failed to download segmentation mask');
      }

      const arrayBuffer = await response.arrayBuffer();

      const header = nifti.readHeader(arrayBuffer);
      const dims = [header.dims[1], header.dims[2], header.dims[3]];
      console.log("🟢 Segmentation dimensions:", dims);

      const dataBuffer = nifti.readImage(header, arrayBuffer);

      let data;
      const datatype = (header as any).datatype || 4;
      if (datatype === 4) {
        const int16Data = new Int16Array(dataBuffer);
        data = new Float32Array(int16Data);
      } else if (datatype === 8) {
        const int32Data = new Int32Array(dataBuffer);
        data = new Float32Array(int32Data);
      } else if (datatype === 16) {
        data = new Float32Array(dataBuffer);
      } else {
        const int16Data = new Int16Array(dataBuffer);
        data = new Float32Array(int16Data);
      }

      // Apply same rotation as original image
      const [x, y, z] = dims;
      const rotated = new Float32Array(data.length);
      for (let k = 0; k < z; k++) {
        for (let j = 0; j < y; j++) {
          for (let i = 0; i < x; i++) {
            const originalIndex = i + j * x + k * x * y;
            const newI = j;
            const newJ = x - 1 - i;
            const newK = k;
            const flippedJ = y - 1 - newJ;
            const newIndex = newI + flippedJ * y + newK * y * x;
            rotated[newIndex] = data[originalIndex];
          }
        }
      }

      setPredictionMask(rotated);
      setPredictionDimensions([y, x, z]);
      console.log('🟢 Segmentation mask loaded successfully');
      
      return rotated;
    } catch (error) {
      console.error('❌ Failed to load segmentation mask:', error);
      throw error;
    }
  };

  const handlePrediction = async () => {
    if (!niftiData || !mriId) {
      alert("Please load an image first before running prediction.");
      return;
    }
    
    setIsPredicting(true);
    setPredictionError(null);
    setPredictionProgress('Initializing...');

    try {
      // Step 1: Detect modality
      setPredictionProgress('Detecting MRI modality...');
      console.log('Step 1: Detecting modality...');
      await detectModality(mriId);

      // Step 2: Run segmentation
      setPredictionProgress(`Running ${selectedModel.toUpperCase()} segmentation model... (this may take 15-20 seconds)`);
      console.log(`Step 2: Running segmentation with ${selectedModel} model...`);
      const segResult = await runSegmentation(mriId, selectedModel);
      
      console.log('Segmentation summary:', segResult.summary);
      setSegmentationSummary(segResult.summary);

      // Step 3: Download and load the mask
      setPredictionProgress('Downloading segmentation results...');
      console.log('Step 3: Downloading segmentation mask...');
      await downloadSegmentationMask(mriId);

      setPredictionProgress('Complete!');
      setShowSuccessModal(true);

    } catch (error: any) {
      console.error('❌ Prediction pipeline failed:', error);
      setPredictionError(error.message);
      alert(`Prediction failed: ${error.message}`);
    } finally {
      setIsPredicting(false);
      setPredictionProgress('');
    }
  };

  const downloadSegmentationAsNifti = async () => {
    if (!mriId) {
      alert("No segmentation data available to download.");
      return;
    }

    try {
      const metadataResponse = await fetch(`${API_BASE_URL}/mri/${mriId}/segmentation`);
      const metadata = await metadataResponse.json();
      const filename = metadata.file_name;
      const response = await fetch(`${API_BASE_URL}/mri/${mriId}/segmentation/data`);
      
      if (!response.ok) {
        throw new Error('Failed to download segmentation file');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      console.log('✅ Segmentation file downloaded successfully:', filename);
    } catch (error) {
      console.error('❌ Failed to download segmentation file:', error);
    }
  };

  // Render a single view to a specific canvas with per-window settings
  const renderViewToCanvas = (
    canvas: HTMLCanvasElement,
    view: 'axial' | 'coronal' | 'sagittal',
    sliceIndex: number,
    winLevel: number,
    winWidth: number,
    zoom: number,
    pan: { x: number; y: number },
    // Global overlay data
    windowShowOverlay: boolean = false,
    windowSegmentationData: Float32Array | null = null,
    windowSegmentationDimensions: [number, number, number] = [0, 0, 0]
  ) => {
    if (!niftiData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn('Cannot get 2D context for canvas - it may already have a WebGL context');
      return;
    }

    const { slice, width, height } = getSlice(niftiData, dimensions, sliceIndex, view);

    // Calculate display size based on canvas container
    const containerWidth = canvas.parentElement?.clientWidth || 400;
    const containerHeight = canvas.parentElement?.clientHeight || 400;
    const scale = Math.min(containerWidth / width, containerHeight / height) * 0.9;
    const displayWidth = Math.floor(width * scale);
    const displayHeight = Math.floor(height * scale);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = containerWidth * dpr;
    canvas.height = containerHeight * dpr;
    canvas.style.width = `${containerWidth}px`;
    canvas.style.height = `${containerHeight}px`;

    // Apply zoom and pan transformations
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, containerWidth * dpr, containerHeight * dpr);
    
    // Fill canvas with pure black first
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, containerWidth * dpr, containerHeight * dpr);
    
    // Apply transformations: DPR first, then zoom and pan
    const centerX = containerWidth / 2;
    const centerY = containerHeight / 2;
    
    // Start with DPR scaling
    ctx.scale(dpr, dpr);
    
    // Apply pan (in screen space)
    ctx.translate(pan.x, pan.y);
    
    // Move to center, apply zoom, move back
    ctx.translate(centerX, centerY);
    ctx.scale(zoom, zoom);
    ctx.translate(-centerX, -centerY);
    
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

    // window/level - use per-window values
    const wlMin = winLevel - winWidth / 2;
    const wlMax = winLevel + winWidth / 2;
    const windowed = new Uint8ClampedArray(width * height);

    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        const val = slice[i + j * width];
        if (val < wlMin) windowed[i + j * width] = 0;
        else if (val > wlMax) windowed[i + j * width] = 255;
        else {
      const normalized = (val - wlMin) / (wlMax - wlMin);
          windowed[i + j * width] = Math.floor(Math.pow(normalized, 0.9) * 255);
        }
      }
    }

    const imageData = ctx.createImageData(displayWidth, displayHeight);

    for (let yPix = 0; yPix < displayHeight; yPix++) {
      for (let xPix = 0; xPix < displayWidth; xPix++) {
        const srcX = xPix / scale;
        const srcY = yPix / scale;
        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        const x2 = Math.min(x1 + 1, width - 1);
        const y2 = Math.min(y1 + 1, height - 1);
        const fx = srcX - x1;
        const fy = srcY - y1;

        const v1 = windowed[x1 + y1 * width] || 0;
        const v2 = windowed[x2 + y1 * width] || 0;
        const v3 = windowed[x1 + y2 * width] || 0;
        const v4 = windowed[x2 + y2 * width] || 0;

        const val = Math.floor(
          v1 * (1 - fx) * (1 - fy) +
          v2 * fx * (1 - fy) +
          v3 * (1 - fx) * fy +
          v4 * fx * fy
        );

        // Get original data value to check if it's background
        const origVal1 = slice[x1 + y1 * width] || 0;
        const origVal2 = slice[x2 + y1 * width] || 0;
        const origVal3 = slice[x1 + y2 * width] || 0;
        const origVal4 = slice[x2 + y2 * width] || 0;
        const origVal =
          origVal1 * (1 - fx) * (1 - fy) +
          origVal2 * fx * (1 - fy) +
          origVal3 * (1 - fx) * fy +
          origVal4 * fx * fy;

        const pixelIndex = (yPix * displayWidth + xPix) * 4;

        // Use fixed absolute threshold for background - independent of window/level settings
        // This ensures background stays black regardless of brightness/contrast adjustments
        const absoluteBackgroundThreshold = 50; // Fixed threshold for background detection
        if (origVal < absoluteBackgroundThreshold) {
          // Set to pure black background
          imageData.data[pixelIndex] = 0;
          imageData.data[pixelIndex + 1] = 0;
          imageData.data[pixelIndex + 2] = 0;
          imageData.data[pixelIndex + 3] = 255;
        } else {
          imageData.data[pixelIndex] = val;
          imageData.data[pixelIndex + 1] = val;
          imageData.data[pixelIndex + 2] = val;
          imageData.data[pixelIndex + 3] = 255;
        }
      }
    }

    // Create temporary canvas for the image data
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = displayWidth;
    tempCanvas.height = displayHeight;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Center the image in the container
    const offsetX = (containerWidth - displayWidth) / 2;
    const offsetY = (containerHeight - displayHeight) / 2;
    
    if (tempCtx) {
      tempCtx.putImageData(imageData, 0, 0);
      
      // Draw the temp canvas onto the main canvas with current transformations
      ctx.drawImage(tempCanvas, offsetX, offsetY, displayWidth, displayHeight);
    }

    if (windowShowOverlay && windowSegmentationData && windowSegmentationDimensions[0] > 0) {
    try {
      const { slice: maskSlice, width: maskWidth, height: maskHeight } = getOverlaySlice(
        windowSegmentationData,
        windowSegmentationDimensions,
        sliceIndex,
        view
      );

      // Create overlay canvas
      const overlayCanvas = document.createElement('canvas');
      overlayCanvas.width = displayWidth;
      overlayCanvas.height = displayHeight;
      const overlayCtx = overlayCanvas.getContext('2d');
      
      if (overlayCtx) {
        const overlayImageData = overlayCtx.createImageData(displayWidth, displayHeight);

        // Render the overlay with color coding
        for (let yPix = 0; yPix < displayHeight; yPix++) {
          for (let xPix = 0; xPix < displayWidth; xPix++) {
            const srcX = xPix / scale;
            const srcY = yPix / scale;
            const x1 = Math.floor(srcX);
            const y1 = Math.floor(srcY);
            const x2 = Math.min(x1 + 1, maskWidth - 1);
            const y2 = Math.min(y1 + 1, maskHeight - 1);
            const fx = srcX - x1;
            const fy = srcY - y1;

            const m1 = maskSlice[x1 + y1 * maskWidth] || 0;
            const m2 = maskSlice[x2 + y1 * maskWidth] || 0;
            const m3 = maskSlice[x1 + y2 * maskWidth] || 0;
            const m4 = maskSlice[x2 + y2 * maskWidth] || 0;

            const maskVal =
              m1 * (1 - fx) * (1 - fy) +
              m2 * fx * (1 - fy) +
              m3 * (1 - fx) * fy +
              m4 * fx * fy;

            const pixelIndex = (yPix * displayWidth + xPix) * 4;

            if (maskVal > 0.5) {
              const alpha = 0.4;
              
              // Color coding: 1=red (necrotic), 2=green (edema), 3=blue (enhancing)
              let r = 0, g = 0, b = 0;
              const roundedMaskVal = Math.round(maskVal);
              
              if (roundedMaskVal === 1) {
                r = 255; g = 0; b = 0; // Red for necrotic
              } else if (roundedMaskVal === 2) {
                r = 0; g = 255; b = 0; // Green for edema
              } else if (roundedMaskVal === 3) {
                r = 0; g = 100; b = 255; // Blue for enhancing
              }

              overlayImageData.data[pixelIndex] = r;
              overlayImageData.data[pixelIndex + 1] = g;
              overlayImageData.data[pixelIndex + 2] = b;
              overlayImageData.data[pixelIndex + 3] = Math.floor(alpha * 255); // Alpha channel
            } else {
              // Transparent where no tumor
              overlayImageData.data[pixelIndex + 3] = 0;
            }
          }
        }

        overlayCtx.putImageData(overlayImageData, 0, 0);
        
        // Draw overlay on main canvas with same offset as the MRI image
        ctx.drawImage(overlayCanvas, offsetX, offsetY, displayWidth, displayHeight);
      }
      } catch (error) {
        console.error('Error rendering overlay:', error);
      }
    }
  };

  // Draw measurements and annotations on overlay for a specific window with zoom/pan transformations
  const drawMeasurementsAndAnnotations = (overlayIdx: number, windowId: number) => {
    const overlay = overlayRefs.current[overlayIdx];
    if (!overlay) return;

    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    // Get the window's zoom and pan state
    const window = dynamicWindows.find(w => w.id === windowId);
    if (!window) return;

    const zoom = window.zoomLevel;
    const pan = window.panOffset;

    const dpr = globalThis.devicePixelRatio || 1;
    
    // Set canvas size to match its display size
    const parent = overlay.parentElement;
    const containerWidth = parent?.clientWidth || 400;
    const containerHeight = parent?.clientHeight || 400;
    
    if (parent) {
      overlay.width = containerWidth * dpr;
      overlay.height = containerHeight * dpr;
      overlay.style.width = `${containerWidth}px`;
      overlay.style.height = `${containerHeight}px`;
    }
    
    // Apply the same transformations as the main canvas to make annotations stick to the image
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    
    // Apply transformations: DPR first, then zoom and pan (same as main canvas)
    const centerX = containerWidth / 2;
    const centerY = containerHeight / 2;
    
    // Start with DPR scaling
    ctx.scale(dpr, dpr);
    
    // Apply pan (in screen space)
    ctx.translate(pan.x, pan.y);
    
    // Move to center, apply zoom, move back
    ctx.translate(centerX, centerY);
    ctx.scale(zoom, zoom);
    ctx.translate(-centerX, -centerY);

    // Draw only measurements for this window
    // Line width and font size should stay constant regardless of zoom
    const baseLineWidth = 2 / zoom;
    const baseFontSize = 12 / zoom;
    const baseCircleRadius = 8 / zoom;
    
    measurements.filter(m => m.windowId === windowId).forEach((m) => {
      ctx.beginPath();
      ctx.moveTo(m.x1, m.y1);
      ctx.lineTo(m.x2, m.y2);
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = baseLineWidth;
      ctx.stroke();
      
          const dx = m.x2 - m.x1;
          const dy = m.y2 - m.y1;
      const distance = Math.sqrt(dx * dx + dy * dy);
      ctx.fillStyle = '#00FF00';
      ctx.font = `${baseFontSize}px system-ui`;
      ctx.fillText(`${distance.toFixed(1)}px`, (m.x1 + m.x2) / 2 + 5 / zoom, (m.y1 + m.y2) / 2 - 5 / zoom);
    });

    // Draw only annotations for this window
    const windowAnnotations = annotations.filter(a => a.windowId === windowId);
    windowAnnotations.forEach((a) => {
      ctx.beginPath();
      ctx.arc(a.x, a.y, baseCircleRadius, 0, Math.PI * 2);
      ctx.fillStyle = a.color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = baseLineWidth;
      ctx.stroke();

      // Draw text if hovered (need to find the actual index in the full annotations array)
      const fullIdx = annotations.indexOf(a);
      if (hoveredAnnotation === fullIdx && a.text) {
        const textOffset = 12 / zoom;
        const boxWidth = 100 / zoom;
        const boxHeight = 24 / zoom;
        const textPadding = 4 / zoom;
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(a.x + textOffset, a.y - 20 / zoom, boxWidth, boxHeight);
        ctx.fillStyle = '#fff';
        ctx.font = `${baseFontSize}px system-ui`;
        ctx.fillText(a.text, a.x + textOffset + textPadding, a.y - textPadding);
      }
    });
  };

  // Render all views to the grid (dynamic windows) - using per-window state
  const renderAllViews = () => {
    if (!niftiData) {
      console.log('renderAllViews: no niftiData');
      return;
    }

    console.log('renderAllViews called for', dynamicWindows.length, 'windows');

    // Render each window based on its view type and individual state
    dynamicWindows.forEach((window, idx) => {
      const canvas = canvasRefs.current[idx];
      if (!canvas || window.view === '3d') return; // Skip 3D windows (handled separately)

      const view = window.view as 'axial' | 'coronal' | 'sagittal';
      
      // Use the window's own slice index
      const sliceIndex = window.slice;

      renderViewToCanvas(
        canvas, 
        view, 
        sliceIndex, 
        window.windowLevel, 
        window.windowWidth,
        window.zoomLevel,
        window.panOffset,
        showOverlay,
        predictionMask,
        predictionDimensions
      );
      
      // Draw measurements and annotations on the overlay for this specific window
      drawMeasurementsAndAnnotations(idx, window.id);
      
      console.log(`Rendered ${view} view in window ${window.id} at slice ${sliceIndex}`);
    });
  };

  // Create a stable reference that only changes when window structure changes (not properties like zoom/brightness)
  const windowStructureKey = dynamicWindows.map(w => `${w.id}-${w.view}`).join(',');
  const windowStructure = useMemo(() => 
    dynamicWindows.map(w => ({ id: w.id, view: w.view })),
    [windowStructureKey]
  );

  // Initialize 3D rendering for any window with view '3d' - MUST happen BEFORE 2D rendering
  useEffect(() => {
    if (!currentFile) return;

    const currentWindowIds = new Set(dynamicWindows.filter(w => w.view === '3d').map(w => w.id));
    
    // Clean up 3D viewers that are no longer needed
    nv3DRefs.current.forEach((nv, windowId) => {
      if (!currentWindowIds.has(windowId)) {
        console.log(`🟢 Cleaning up 3D viewer for window ${windowId}`);
        try {
          nv.destroy();
          nv3DRefs.current.delete(windowId);
        } catch (e) {
          console.error('Error destroying 3D viewer:', e);
        }
      }
    });

    // Initialize 3D rendering for each window with 3D view (with slight delay to ensure canvas is ready)
    const timer = setTimeout(() => {
      dynamicWindows.forEach((window, idx) => {
        if (window.view !== '3d') return;

        const canvas = canvasRefs.current[idx];
        if (!canvas) {
          console.log(`3D render: canvas not ready for window ${window.id}`);
          return;
        }

        // Skip if already initialized for this window
        if (nv3DRefs.current.has(window.id)) {
          console.log(`3D viewer already exists for window ${window.id}`);
          return;
        }

        console.log(`🟢 Initializing 3D render view in window ${window.id}`);
        
        // Ensure canvas has proper dimensions
        const parent = canvas.parentElement;
        if (parent) {
          canvas.width = parent.clientWidth;
          canvas.height = parent.clientHeight;
          canvas.style.width = '100%';
          canvas.style.height = '100%';
          console.log("Canvas dimensions set:", canvas.width, "x", canvas.height);
        }

        const nv = new Niivue({
          show3Dcrosshair: false,
          backColor: [0, 0, 0, 1],
          isOrientCube: true,
        });

        nv.attachToCanvas(canvas);
        nv3DRefs.current.set(window.id, nv);

        const url = URL.createObjectURL(currentFile);
        console.log("🟢 Loading volume into 3D viewer...");
        nv.loadVolumes([{ 
          url, 
          name: currentFile.name,
          colormap: 'gray',
          opacity: 1.0,
        }]).then(() => {
          console.log("🟢 Volume loaded, setting render mode");
          nv.setSliceType(nv.sliceTypeRender); // Set to 3D render mode
          
          console.log("🟢 3D render view initialized successfully");
        }).catch((error) => {
          console.error("❌ Error loading volume:", error);
        });
      });
    }, 100); // Small delay to ensure canvas is in DOM and ready

    return () => {
      clearTimeout(timer); // Clear the timeout
      // Cleanup all 3D viewers on unmount
      nv3DRefs.current.forEach((nv) => {
        try {
          nv.destroy();
        } catch (e) {
          console.error('Error cleaning up 3D viewer:', e);
        }
      });
      nv3DRefs.current.clear();
    };
  }, [currentFile, windowStructure]);

  // Render 2D views - happens AFTER 3D initialization
  useEffect(() => {
    renderAllViews();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [niftiData, flipH, flipV, measurements, annotations, mode, hoveredAnnotation, cursorPosition, dynamicWindows, showOverlay, predictionMask]);

  // Cleanup viewer windows on unmount
  useEffect(() => {
    return () => {
      viewerWindows.forEach(window => {
        if (window.nv) {
          try {
            window.nv.destroy();
          } catch (e) {
            console.error('Error destroying viewer:', e);
          }
        }
      });
    };
  }, [viewerWindows]);

  // file input handler
  useEffect(() => {
    const input = document.getElementById('niftiUpload') as HTMLInputElement | null;
    console.log('Setting up file upload handler. Input element found:', !!input);
    if (!input) return;
    const handleFileChange = async (e: Event) => {
      console.log('File input changed!');
      const file = (e.target as HTMLInputElement).files?.[0];
      console.log('Selected file:', file?.name);
    if (!file) return;
    if (file.name.toLowerCase().endsWith('.nii') || file.name.toLowerCase().endsWith('.nii.gz')) {
        console.log('Valid NIfTI file detected, loading...');
      await loadNiftiFile(file);
    } else {
        console.error('Please upload a .nii or .nii.gz file');
      }
    };
    input.addEventListener('change', handleFileChange);
    return () => input.removeEventListener('change', handleFileChange);
  }, [expandedSections.upload]); // Re-run when upload section expands

  // Check for uploaded file from landing page or demo mode
  useEffect(() => {
    const state = location.state as { uploadedFile?: File; demoMode?: boolean } | null;

    // Handle uploaded file
    if (state?.uploadedFile) {
      const file = state.uploadedFile;
      console.log("File received from landing page:", file.name);
      if (file.name.toLowerCase().endsWith('.nii') || file.name.toLowerCase().endsWith('.nii.gz')) {
        loadNiftiFile(file);
      } else {
        console.error('Please upload a .nii or .nii.gz file');
      }
    }

    // Handle demo mode - load sample file (only once)
    else if (state?.demoMode && !demoLoadedRef.current) {
      console.log("Demo mode activated - loading sample file");
      demoLoadedRef.current = true;
      loadDemoFile();
    }
  }, [location]);

  // Load demo NIfTI file from public folder
  const loadDemoFile = async () => {
    try {
      console.log("Loading demo files...");
      
      // Define all 4 modality files
      const modalities = ['flair', 't1', 't1ce', 't2'];
      const patientId = 'BraTS20_Validation_001';
      
      // Try to use backend, fallback to direct loading if backend unavailable
      let backendAvailable = true;
      let uploadedMriId: number | null = null;
      
      try {
        // First, check if demo files already exist in backend
        console.log("Checking if demo files already exist in backend...");
        const checkResponse = await fetch(`${API_BASE_URL}/mri`);
        
        if (checkResponse.ok) {
          const existingFiles = await checkResponse.json();
          
          // Look for existing flair file for this patient
          const existingFlair = existingFiles.find((file: any) => 
            file.file_name === `${patientId}_flair.nii`
          );
          
          if (existingFlair) {
            console.log("🟢 Demo files already exist in backend (S3), loading directly from backend...");
            uploadedMriId = existingFlair.id;
            setMriId(existingFlair.id);
            
            // Fetch the flair file directly from backend (which gets it from S3)
            console.log(`Fetching flair file from backend (MRI ID: ${existingFlair.id})...`);
            const dataResponse = await fetch(`${API_BASE_URL}/mri/${existingFlair.id}/data`);
            
            if (!dataResponse.ok) {
              throw new Error(`Failed to fetch flair file from backend: ${dataResponse.statusText}`);
            }
            
            const flairArrayBuffer = await dataResponse.arrayBuffer();
            const flairBlob = new Blob([flairArrayBuffer]);
            const flairFile = new File([flairBlob], `${patientId}_flair.nii`, { type: 'application/octet-stream' });
            
            // Skip backend upload since file is already in S3!
            await loadNiftiFile(flairFile, true);
            
            console.log("✅ Demo file loaded successfully from backend (S3) - FAST!");
            console.log(`🟢 Prediction features enabled with MRI ID: ${uploadedMriId}`);
            return; // Exit early, we're done!
            
          } else {
            // Files don't exist, need to upload them
            console.log("Demo files not found in backend, uploading...");
            
            for (const modality of modalities) {
              const filename = `${patientId}_${modality}.nii`;
              console.log(`Fetching ${filename}...`);
              
              const response = await fetch(`/${filename}`);
      if (!response.ok) {
                throw new Error(`Demo file not found: ${filename}. Please add all 4 modality files to the public folder.`);
      }
              
      const arrayBuffer = await response.arrayBuffer();
      const blob = new Blob([arrayBuffer]);
              const file = new File([blob], filename, { type: 'application/octet-stream' });
              
              // Upload each file to backend
              console.log(`Uploading ${filename} to backend...`);
              const uploadResult = await uploadNiftiFile(file);
              
              // Store the first MRI ID (flair) for prediction
              if (modality === 'flair' && uploadResult.mri_id) {
                uploadedMriId = uploadResult.mri_id;
                console.log(`🟢 Stored MRI ID: ${uploadedMriId} for prediction`);
              }
            }
            
            console.log("All 4 modality files uploaded successfully to backend");
            
            // Set the mriId for prediction features
            if (uploadedMriId) {
              setMriId(uploadedMriId);
            }
          }
        }
        
      } catch (backendError) {
        console.warn("Backend not available, using direct file loading:", backendError);
        backendAvailable = false;
      }
      
      // Load the flair file for display (only if we didn't already load from backend)
      if (!uploadedMriId || !backendAvailable) {
        console.log("Loading flair file from public folder...");
        const flairResponse = await fetch(`/${patientId}_flair.nii`);
        if (!flairResponse.ok) {
          throw new Error(`Demo file not found: ${patientId}_flair.nii. Please add the file to the public folder.`);
        }
        
        const flairArrayBuffer = await flairResponse.arrayBuffer();
        const flairBlob = new Blob([flairArrayBuffer]);
        const flairFile = new File([flairBlob], `${patientId}_flair.nii`, { type: 'application/octet-stream' });
        
        await loadNiftiFile(flairFile);
        
        if (backendAvailable) {
          console.log("🟢 Demo files loaded successfully with backend support");
          console.log(`🟢 Prediction features enabled with MRI ID: ${uploadedMriId}`);
        } else {
          console.log("🟢 Demo files loaded successfully (backend unavailable - prediction features disabled)");
        }
      }
    } catch (error) {
      console.error("Failed to load demo files:", error);
      alert(`Demo files not found. Please make sure the flair file is in the public folder:\n- BraTS20_Validation_001_flair.nii`);
    }
  };

  // Close a viewer window
  const closeViewerWindow = (id: string) => {
    setViewerWindows(prev => {
      const window = prev.find(w => w.id === id);
      if (window?.nv) {
        try {
          window.nv.destroy();
        } catch (e) {
          console.error('Error destroying viewer:', e);
        }
      }
      return prev.filter(w => w.id !== id);
    });
  };

  const loadFileIntoWindow = (windowId: string, file: File) => {
    setViewerWindows(prev =>
      prev.map(w => (w.id === windowId ? { ...w, file, title: file.name } : w))
    );
  };

  // Update window position (for dragging)
  const updateWindowPosition = (id: string, x: number, y: number) => {
    setViewerWindows(prev => prev.map(w =>
      w.id === id ? { ...w, position: { x, y } } : w
    ));
  };

  // Toggle minimize window
  const toggleMinimizeWindow = (id: string) => {
    setViewerWindows(prev => prev.map(w =>
      w.id === id ? { ...w, isMinimized: !w.isMinimized } : w
    ));
  };

// Load current file into window
  const loadCurrentFileIntoWindow = (windowId: string) => {
    const state = location.state as { uploadedFile?: File } | null;
    if (state?.uploadedFile) {
      loadFileIntoWindow(windowId, state.uploadedFile);
    }
  };


  const DraggableViewerWindow: React.FC<{
    window: typeof viewerWindows[0];
    onClose: () => void;
    onMinimize: () => void;
    onLoadFile: (file: File) => void;
    onLoadCurrent: () => void;
    onPositionChange: (x: number, y: number) => void;
  }> = ({
    window: win,
    onClose,
    onMinimize,
    onLoadFile,
    onLoadCurrent,
    onPositionChange,
  }) => {
    const [isDragging, setIsDragging] = useState(false);
    const dragOffset = useRef({ x: 0, y: 0 });
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [showCrosshair, setShowCrosshair] = useState(false);

    useEffect(() => {
      if (!win.file || !win.canvasRef.current) return;

      const nv = new Niivue({
        show3Dcrosshair: showCrosshair,
        backColor: [0, 0, 0, 1],
      });

      nv.attachToCanvas(win.canvasRef.current);

      const url = URL.createObjectURL(win.file);
      nv.loadVolumes([{ url, name: win.file.name }]).then(() => {
        nv.setSliceType(nv.sliceTypeAxial);
      });

      win.nv = nv;

      return () => {
        URL.revokeObjectURL(url);
      };
    }, [win.file, showCrosshair]);

    const handleMouseDown = (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest(".window-controls")) return;
      dragOffset.current = {
        x: e.clientX - win.position.x,
        y: e.clientY - win.position.y,
      };
      setIsDragging(true);
    };

    useEffect(() => {
      if (!isDragging) return;

      const handleMouseMove = (e: MouseEvent) => {
        onPositionChange(
          e.clientX - dragOffset.current.x,
          e.clientY - dragOffset.current.y
        );
      };
      const handleMouseUp = () => setIsDragging(false);

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }, [isDragging, onPositionChange]);

    const toggleCrosshair = () => {
      if (win.nv) {
        const newState = !showCrosshair;
        setShowCrosshair(newState);
        win.nv.opts.show3Dcrosshair = newState;
        win.nv.drawScene();
      }
    };

    const setView = (type: string) => {
      if (!win.nv) return;
      switch (type) {
        case "axial":
          win.nv.setSliceType(win.nv.sliceTypeAxial);
          break;
        case "coronal":
          win.nv.setSliceType(win.nv.sliceTypeCoronal);
          break;
        case "sagittal":
          win.nv.setSliceType(win.nv.sliceTypeSagittal);
          break;
        case "render":
          win.nv.setSliceType(win.nv.sliceTypeRender);
          break;
        default:
          win.nv.setSliceType(win.nv.sliceTypeMultiplanar);
      }
    };

    return (
      <div
        style={{
          position: "fixed",
          left: win.position.x,
          top: win.position.y,
          width: win.size.width,
          height: win.isMinimized ? "auto" : win.size.height,
          background: "rgba(30, 41, 59, 0.98)",
          border: "1px solid rgba(148, 163, 184, 0.3)",
          borderRadius: "12px",
          boxShadow: "0 8px 32px 0 rgba(0, 0, 0, 0.5)",
          zIndex: 100,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          userSelect: isDragging ? "none" : "auto",
        }}
      >
        {/* === Title Bar === */}
        <div
          onMouseDown={handleMouseDown}
          style={{
            padding: "12px 16px",
            background: "rgba(51, 65, 85, 0.6)",
            cursor: isDragging ? "grabbing" : "grab",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div
            style={{
              fontSize: "14px",
              fontWeight: 600,
              color: "#f1f5f9",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: 1,
            }}
          >
            {win.title}
          </div>

          <div className="window-controls" style={{ display: "flex", gap: "8px" }}>
            <button onClick={onMinimize} title="Minimize">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
            </button>
            <button onClick={onClose} title="Close">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </div>

        {/* === Viewer Content === */}
        {!win.isMinimized && (
          <>
            <div
              style={{
                flex: 1,
                background: "#1a1a1a",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
              }}
            >
              {!win.file && (
                <div
                  style={{
                    position: "absolute",
                    color: "#94a3b8",
                    textAlign: "center",
                    padding: "20px",
                  }}
                >
                  <p>No file loaded</p>
                  <p style={{ fontSize: "12px", marginTop: "8px" }}>
                    Use the controls below to load a file
                  </p>
                </div>
              )}

              <canvas
                ref={win.canvasRef}
                style={{
                  width: "100%",
                  height: "100%",
                  display: "block",
                }}
              />
            </div>

            {/* === Controls === */}
            <div
              style={{
                padding: "12px",
                background: "rgba(51, 65, 85, 0.4)",
                borderTop: "1px solid rgba(148, 163, 184, 0.2)",
                display: "flex",
                flexWrap: "wrap",
                gap: "8px",
                justifyContent: "space-between",
              }}
            >
              <div style={{ display: "flex", gap: "8px" }}>
                <button onClick={onLoadCurrent}>Load Current File</button>
                <button onClick={() => fileInputRef.current?.click()}>
                  Load Different File
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".nii,.nii.gz"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onLoadFile(file);
                  }}
                />
              </div>

              {/* View and Crosshair Controls */}
              <div style={{ display: "flex", gap: "6px" }}>
                <button onClick={() => setView("axial")}>Axial</button>
                <button onClick={() => setView("coronal")}>Coronal</button>
                <button onClick={() => setView("sagittal")}>Sagittal</button>
                <button onClick={() => setView("render")}>3D</button>
                <button onClick={toggleCrosshair}>
                  {showCrosshair ? "Hide Crosshair" : "Show Crosshair"}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    );
  };



  // overlay events for measure/annotate - work with selected window
  useEffect(() => {
    if (mode === 'none') return; // Only attach when a measurement mode is active
    
    // Find the index of the selected window
    const selectedWindowIndex = dynamicWindows.findIndex(w => w.id === selectedWindowId);
    if (selectedWindowIndex === -1) return;
    
    const overlay = overlayRefs.current[selectedWindowIndex];
    if (!overlay) return;
    const rect = () => overlay.getBoundingClientRect();

    const toLocal = (ev: MouseEvent) => {
      const r = rect();
      return { x: ev.clientX - r.left, y: ev.clientY - r.top };
    };

    const onDown = (ev: MouseEvent) => {
      const p = toLocal(ev);

      // DEBUG: Log click coordinates and canvas dimensions
      const r = rect();
      console.log('Click Debug:', {
        clientX: ev.clientX,
        clientY: ev.clientY,
        overlayLeft: r.left,
        overlayTop: r.top,
        overlayWidth: r.width,
        overlayHeight: r.height,
        localX: p.x,
        localY: p.y,
        mode: mode
      });

      if (mode === 'erase') {
        // Transform mouse position to match annotation/measurement coordinate space
        const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
        if (selectedWindow) {
          const zoom = selectedWindow.zoomLevel;
          const pan = selectedWindow.panOffset;
          const parent = overlay.parentElement;
          const containerWidth = parent?.clientWidth || 400;
          const containerHeight = parent?.clientHeight || 400;
          const centerX = containerWidth / 2;
          const centerY = containerHeight / 2;
          
          // Reverse the transformations to get annotation space coordinates
          const transformedX = ((p.x - pan.x - centerX) / zoom) + centerX;
          const transformedY = ((p.y - pan.y - centerY) / zoom) + centerY;
          
          // Check if clicking on an annotation (only for selected window)
        let annotationRemoved = false;
        annotations.forEach((a, idx) => {
            if (a.windowId !== selectedWindowId) return; // Only check annotations for selected window
            const dx = transformedX - a.x;
            const dy = transformedY - a.y;
          const distance = Math.sqrt(dx*dx + dy*dy);
          if (distance <= 12 && !annotationRemoved) {
            setAnnotations(prev => prev.filter((_, i) => i !== idx));
            annotationRemoved = true;
          }
        });

          // Check if clicking on a measurement (only for selected window)
        if (!annotationRemoved) {
          measurements.forEach((m, idx) => {
              if (m.windowId !== selectedWindowId) return; // Only check measurements for selected window
            // Calculate distance from point to line segment
            const dx = m.x2 - m.x1;
            const dy = m.y2 - m.y1;
            const length = Math.sqrt(dx*dx + dy*dy);
            if (length === 0) return;

              const t = Math.max(0, Math.min(1, ((transformedX - m.x1) * dx + (transformedY - m.y1) * dy) / (length * length)));
            const projX = m.x1 + t * dx;
            const projY = m.y1 + t * dy;
              const distToLine = Math.sqrt((transformedX - projX)**2 + (transformedY - projY)**2);

            if (distToLine <= 8) {
              setMeasurements(prev => prev.filter((_, i) => i !== idx));
            }
          });
          }
        }
      } else if (mode === 'measure' || mode === 'annotate'){
        // Transform mouse coordinates to image space for storing
        const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
        if (selectedWindow) {
          const zoom = selectedWindow.zoomLevel;
          const pan = selectedWindow.panOffset;
          const parent = overlay.parentElement;
          const containerWidth = parent?.clientWidth || 400;
          const containerHeight = parent?.clientHeight || 400;
          const centerX = containerWidth / 2;
          const centerY = containerHeight / 2;
          
          // Transform to image space coordinates
          const transformedX = ((p.x - pan.x - centerX) / zoom) + centerX;
          const transformedY = ((p.y - pan.y - centerY) / zoom) + centerY;
          
          drawingRef.current = { drawing: true, x: transformedX, y: transformedY };
        }
      }
    };

    const onMove = (ev: MouseEvent) => {
      const p = toLocal(ev);

      // Update cursor position for eraser
      if (mode === 'erase') {
        setCursorPosition(p);
      } else {
        setCursorPosition(null);
      }

      // Check if hovering over an annotation (works in all modes except erase, only for selected window)
      if (mode !== 'erase') {
        const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
        if (selectedWindow) {
          const zoom = selectedWindow.zoomLevel;
          const pan = selectedWindow.panOffset;
          const parent = overlay.parentElement;
          const containerWidth = parent?.clientWidth || 400;
          const containerHeight = parent?.clientHeight || 400;
          const centerX = containerWidth / 2;
          const centerY = containerHeight / 2;
          
          // Transform mouse position to match annotation coordinate space
          // Reverse the transformations: un-center, un-zoom, un-pan
          const transformedX = ((p.x - pan.x - centerX) / zoom) + centerX;
          const transformedY = ((p.y - pan.y - centerY) / zoom) + centerY;
          
        let foundHover = false;
        annotations.forEach((a, idx) => {
            if (a.windowId !== selectedWindowId) return; // Only check annotations for selected window
            const dx = transformedX - a.x;
            const dy = transformedY - a.y;
          const distance = Math.sqrt(dx*dx + dy*dy);
          if (distance <= 8) {
            setHoveredAnnotation(idx);
            foundHover = true;
          }
        });
        if (!foundHover && hoveredAnnotation !== null) {
          setHoveredAnnotation(null);
          }
        }
      }

      if (!drawingRef.current) return;

      if (mode === 'measure'){
        // live preview: redraw overlays by calling renderAllViews (which redraws existing measurements with transformations)
        renderAllViews();
        const octx = overlay.getContext('2d');
        if (!octx) return;
        
        // Get the selected window's zoom and pan
        const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
        if (!selectedWindow) return;
        const zoom = selectedWindow.zoomLevel;
        const pan = selectedWindow.panOffset;
        
        const dpr = window.devicePixelRatio || 1;
        const parent = overlay.parentElement;
        const containerWidth = parent?.clientWidth || 400;
        const containerHeight = parent?.clientHeight || 400;
        
        // Apply the same transformations as the main canvas
        const centerX = containerWidth / 2;
        const centerY = containerHeight / 2;
        
        octx.setTransform(1, 0, 0, 1, 0, 0);
        octx.scale(dpr, dpr);
        octx.translate(pan.x, pan.y);
        octx.translate(centerX, centerY);
        octx.scale(zoom, zoom);
        octx.translate(-centerX, -centerY);
        
        // Transform current mouse position to image space
        const transformedX = ((p.x - pan.x - centerX) / zoom) + centerX;
        const transformedY = ((p.y - pan.y - centerY) / zoom) + centerY;
        
        // Draw preview line (in image space coordinates)
        octx.beginPath();
        octx.moveTo(drawingRef.current.x, drawingRef.current.y);
        octx.lineTo(transformedX, transformedY);
        octx.strokeStyle = '#00FF00';
        octx.lineWidth = 2 / zoom; // Keep line width constant
        octx.stroke();
        
        const dx = transformedX - drawingRef.current.x;
        const dy = transformedY - drawingRef.current.y;
        octx.fillStyle = '#00FF00';
        octx.font = `${12 / zoom}px system-ui`; // Keep font size constant
        octx.fillText(`${Math.sqrt(dx*dx + dy*dy).toFixed(1)}px`, (drawingRef.current.x + transformedX)/2 + 5 / zoom, (drawingRef.current.y + transformedY)/2 - 5 / zoom);
      } else if (mode === 'annotate'){
        renderAllViews();
        const octx = overlay.getContext('2d');
        if (!octx) return;
        
        // Get the selected window's zoom and pan
        const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
        if (!selectedWindow) return;
        const zoom = selectedWindow.zoomLevel;
        const pan = selectedWindow.panOffset;
        
        const dpr = window.devicePixelRatio || 1;
        const parent = overlay.parentElement;
        const containerWidth = parent?.clientWidth || 400;
        const containerHeight = parent?.clientHeight || 400;
        
        // Apply the same transformations as the main canvas
        const centerX = containerWidth / 2;
        const centerY = containerHeight / 2;
        
        octx.setTransform(1, 0, 0, 1, 0, 0);
        octx.scale(dpr, dpr);
        octx.translate(pan.x, pan.y);
        octx.translate(centerX, centerY);
        octx.scale(zoom, zoom);
        octx.translate(-centerX, -centerY);
        
        // Transform current mouse position to image space
        const transformedX = ((p.x - pan.x - centerX) / zoom) + centerX;
        const transformedY = ((p.y - pan.y - centerY) / zoom) + centerY;
        
        // Draw preview circle (in image space coordinates)
        octx.beginPath();
        octx.arc(transformedX, transformedY, 8 / zoom, 0, Math.PI*2); // Keep circle size constant
        octx.fillStyle = '#FF8C00'; // Orange
        octx.fill();
        octx.strokeStyle = '#fff';
        octx.lineWidth = 2 / zoom; // Keep line width constant
        octx.stroke();
      }
    };

    const onUp = (ev: MouseEvent) => {
      const ref = drawingRef.current;
      if (!ref) return;
      const p = toLocal(ev);
      
      // Transform mouse coordinates to image space
      const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
      if (!selectedWindow) return;
      
      const zoom = selectedWindow.zoomLevel;
      const pan = selectedWindow.panOffset;
      const parent = overlay.parentElement;
      const containerWidth = parent?.clientWidth || 400;
      const containerHeight = parent?.clientHeight || 400;
      const centerX = containerWidth / 2;
      const centerY = containerHeight / 2;
      
      const transformedX = ((p.x - pan.x - centerX) / zoom) + centerX;
      const transformedY = ((p.y - pan.y - centerY) / zoom) + centerY;

      if (mode === 'measure') {
        setMeasurements(ms => [
          ...ms,
          { x1: ref.x, y1: ref.y, x2: transformedX, y2: transformedY, windowId: selectedWindowId },
        ]);
      } else if (mode === 'annotate') {
        // Always use orange color for annotations
        const color = '#FF8C00'; // Orange

        // Show modal for annotation text
        setPendingAnnotation({ x: transformedX, y: transformedY, color });
        setAnnotationText('');
        setShowAnnotationModal(true);
      }

      drawingRef.current = null;
      renderAllViews();
    };

    const onLeave = () => {
      setCursorPosition(null);
    };

    overlay.addEventListener('mousedown', onDown);
    overlay.addEventListener('mousemove', onMove);
    overlay.addEventListener('mouseleave', onLeave);
    window.addEventListener('mouseup', onUp);

    return () => {
      overlay.removeEventListener('mousedown', onDown);
      overlay.removeEventListener('mousemove', onMove);
      overlay.removeEventListener('mouseleave', onLeave);
      window.removeEventListener('mouseup', onUp);
    };
  }, [mode, measurements, annotations, hoveredAnnotation, selectedWindowId, dynamicWindows]);

  // Set cursor based on active mode
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    // Set custom cursors for each mode
    if (mode === 'measure') {
      wrapper.style.cursor = 'crosshair';
    } else if (mode === 'annotate') {
      wrapper.style.cursor = 'cell'; // Plus sign cursor for placing annotations
    } else if (mode === 'erase') {
      wrapper.style.cursor = 'not-allowed'; // X/delete cursor
    } else if (interactionMode === 'pan') {
      wrapper.style.cursor = 'grab';
    } else {
      wrapper.style.cursor = 'default';
    }
  }, [mode, interactionMode]);

  // Interaction mode handlers (scroll, zoom, pan) - now affects only selected window
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || !hasImage) return;

    const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
    if (!selectedWindow) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();

      if (interactionMode === 'scroll') {
        // Scroll through slices in selected window only - use functional update
        setDynamicWindows(prev => prev.map(w => {
          if (w.id !== selectedWindowId) return w;
          
          const view = w.view;
          let maxSlice = 0;
          if (view === 'axial') maxSlice = dimensions[2] - 1;
          else if (view === 'coronal') maxSlice = dimensions[1] - 1;
          else if (view === 'sagittal') maxSlice = dimensions[0] - 1;

          const newSlice = e.deltaY < 0 
            ? Math.max(0, w.slice - 1)
            : Math.min(maxSlice, w.slice + 1);
          
          return { ...w, slice: newSlice };
        }));
      } else if (interactionMode === 'zoom') {
        // Zoom in/out in selected window only - use functional update
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        setDynamicWindows(prev => prev.map(w =>
          w.id === selectedWindowId
            ? { ...w, zoomLevel: Math.max(0.5, Math.min(5, w.zoomLevel + delta)) }
            : w
        ));
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      if (interactionMode === 'pan' || interactionMode === 'brightness' || interactionMode === 'contrast') {
        isPanning.current = true;
        lastPanPos.current = { x: e.clientX, y: e.clientY };
        if (interactionMode === 'pan') {
          wrapper.style.cursor = 'grabbing';
        }
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (isPanning.current) {
          const dx = e.clientX - lastPanPos.current.x;
          const dy = e.clientY - lastPanPos.current.y;
          lastPanPos.current = { x: e.clientX, y: e.clientY };
        
        if (interactionMode === 'pan') {
          // Pan in selected window only - use functional update to get current state
          setDynamicWindows(prev => prev.map(w => 
            w.id === selectedWindowId 
              ? { 
                  ...w, 
                  panOffset: { 
                    x: w.panOffset.x + dx, 
                    y: w.panOffset.y + dy 
                  } 
                }
              : w
          ));
        } else if (interactionMode === 'brightness') {
          // Adjust brightness in selected window only - use functional update
          setDynamicWindows(prev => prev.map(w =>
            w.id === selectedWindowId
              ? { ...w, windowLevel: Math.max(-1000, Math.min(1000, w.windowLevel - dy * 2)) }
              : w
          ));
        } else if (interactionMode === 'contrast') {
          // Adjust contrast in selected window only - use functional update
          setDynamicWindows(prev => prev.map(w =>
            w.id === selectedWindowId
              ? { ...w, windowWidth: Math.max(100, Math.min(2000, w.windowWidth - dy * 2)) }
              : w
          ));
        }
      }
    };

    const handleMouseUp = () => {
      if (interactionMode === 'pan' || interactionMode === 'brightness' || interactionMode === 'contrast') {
        isPanning.current = false;
        wrapper.style.cursor = interactionMode === 'pan' ? 'grab' : 'default';
      }
    };

    wrapper.addEventListener('wheel', handleWheel, { passive: false });
    wrapper.addEventListener('mousedown', handleMouseDown);
    wrapper.addEventListener('mousemove', handleMouseMove);
    wrapper.addEventListener('mouseup', handleMouseUp);
    wrapper.addEventListener('mouseleave', handleMouseUp);

    // Set cursor based on mode
    wrapper.style.cursor = interactionMode === 'pan' ? 'grab' : 'default';

    return () => {
      wrapper.removeEventListener('wheel', handleWheel);
      wrapper.removeEventListener('mousedown', handleMouseDown);
      wrapper.removeEventListener('mousemove', handleMouseMove);
      wrapper.removeEventListener('mouseup', handleMouseUp);
      wrapper.removeEventListener('mouseleave', handleMouseUp);
    };
  }, [interactionMode, hasImage, selectedWindowId, dynamicWindows, dimensions]);

  const resetView = () => {
    // Reset global UI states
    setFlipH(false);
    setFlipV(false);
    setMeasurements([]);
    setAnnotations([]);
    setMode('none');
    setInteractionMode(null);
    setHoveredAnnotation(null);
    
    // Reset the selected window's state
    const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
    if (selectedWindow) {
      const view = selectedWindow.view;
      let defaultSlice = 0;
      if (view === 'axial') defaultSlice = Math.floor(dimensions[2] / 2);
      else if (view === 'coronal') defaultSlice = Math.floor(dimensions[1] / 2);
      else if (view === 'sagittal') defaultSlice = Math.floor(dimensions[0] / 2);
      
      updateWindowState(selectedWindowId, {
        slice: defaultSlice,
        zoomLevel: 1,
        panOffset: { x: 0, y: 0 },
        windowLevel: 200,
        windowWidth: 600
      });
    }
  };

  // Handle annotation modal
  const handleAnnotationSubmit = () => {
    if (pendingAnnotation && annotationText.trim()) {
      setAnnotations(a => [...a, {
        x: pendingAnnotation.x,
        y: pendingAnnotation.y,
        text: annotationText.trim(),
        color: pendingAnnotation.color,
        windowId: selectedWindowId
      }]);
    }
    setShowAnnotationModal(false);
    setPendingAnnotation(null);
    setAnnotationText('');
  };

  const handleAnnotationCancel = () => {
    setShowAnnotationModal(false);
    setPendingAnnotation(null);
    setAnnotationText('');
  };

  // State for prediction mask and overlay visibility
  //const [predictionMask, setPredictionMask] = useState<Float32Array | null>(null);
  //const [showOverlay, setShowOverlay] = useState(false);

  return (
    <div className="viewer-container">
      {/* Success Modal for Prediction */}
      {showSuccessModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(8px)',
            WebkitBackdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001,
          }}
          onClick={() => setShowSuccessModal(false)}
        >
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border: '1px solid rgba(34, 197, 94, 0.3)',
              borderRadius: '16px',
              padding: '32px',
              minWidth: '500px',
              maxWidth: '600px',
              boxShadow: '0 20px 60px 0 rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(34, 197, 94, 0.1)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Success Icon */}
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
              <div style={{
                width: '80px',
                height: '80px',
                margin: '0 auto',
                background: 'rgba(34, 197, 94, 0.1)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '3px solid rgba(34, 197, 94, 0.3)',
              }}>
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="rgb(34, 197, 94)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              </div>
            </div>

            <h3 style={{
              color: '#fff',
              fontSize: '24px',
              fontWeight: 600,
              marginBottom: '12px',
              textAlign: 'center',
            }}>
              Segmentation Complete!
            </h3>
            
            <p style={{
              color: 'rgba(255, 255, 255, 0.7)',
              fontSize: '14px',
              marginBottom: '24px',
              textAlign: 'center',
              lineHeight: '1.6',
            }}>
              Brain tumor segmentation has been successfully completed using the <strong style={{ color: 'rgb(59, 130, 246)' }}>{selectedModel.toUpperCase()}</strong> model. You can now toggle the overlay to view the results.
            </p>

            {/* Summary Statistics */}
            {segmentationSummary && (
              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '24px',
              }}>
                <h4 style={{ color: '#fff', fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>
                  Segmentation Summary
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '13px' }}>
                  <div>
                    <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>Total Tumor Voxels:</span>
                    <div style={{ color: '#fff', fontWeight: 600, marginTop: '4px' }}>
                      {segmentationSummary.total_tumor_voxels?.toLocaleString() || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>Necrotic Core:</span>
                    <div style={{ color: '#fff', fontWeight: 600, marginTop: '4px' }}>
                      {segmentationSummary.necrotic_voxels?.toLocaleString() || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>Edema:</span>
                    <div style={{ color: '#fff', fontWeight: 600, marginTop: '4px' }}>
                      {segmentationSummary.edema_voxels?.toLocaleString() || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>Enhancing Tumor:</span>
                    <div style={{ color: '#fff', fontWeight: 600, marginTop: '4px' }}>
                      {segmentationSummary.enhancing_voxels?.toLocaleString() || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <button
              onClick={downloadSegmentationAsNifti}
              style={{
                width: '100%',
                flex: 1,
                padding: '12px',
                background: 'rgba(59, 130, 246, 0.2)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                borderRadius: '8px',
                color: 'rgb(59, 130, 246)',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                marginBottom: '8px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(59, 130, 246, 0.3)';
                e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.6)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(59, 130, 246, 0.2)';
                e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.4)';
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              Download as NIFTI File
            </button>
            <button
              onClick={() => setShowSuccessModal(false)}
              style={{
                width: '100%',
                padding: '12px',
                background: 'rgba(34, 197, 94, 0.2)',
                border: '1px solid rgba(34, 197, 94, 0.4)',
                borderRadius: '8px',
                color: 'rgb(34, 197, 94)',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(34, 197, 94, 0.3)';
                e.currentTarget.style.borderColor = 'rgba(34, 197, 94, 0.6)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(34, 197, 94, 0.2)';
                e.currentTarget.style.borderColor = 'rgba(34, 197, 94, 0.4)';
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Annotation Modal */}
      {showAnnotationModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            backdropFilter: 'blur(4px)',
            WebkitBackdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={handleAnnotationCancel}
        >
          <div
            style={{
              background: 'rgba(30, 41, 59, 0.95)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border: '1px solid rgba(148, 163, 184, 0.3)',
              borderRadius: '12px',
              padding: '24px',
              minWidth: '400px',
              boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.5)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{
              margin: '0 0 16px 0',
              fontSize: '18px',
              fontWeight: 600,
              color: '#f1f5f9',
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
            }}>Add Annotation</h3>

            <input
              type="text"
              value={annotationText}
              onChange={(e) => setAnnotationText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAnnotationSubmit();
                if (e.key === 'Escape') handleAnnotationCancel();
              }}
              placeholder="Enter annotation text..."
              autoFocus
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '14px',
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(148, 163, 184, 0.3)',
                borderRadius: '8px',
                color: '#f1f5f9',
                outline: 'none',
                marginBottom: '16px',
                boxSizing: 'border-box',
              }}
            />

            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'flex-end'
            }}>
              <button
                onClick={handleAnnotationCancel}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: 500,
                  background: 'rgba(51, 65, 85, 0.5)',
                  border: '1px solid rgba(148, 163, 184, 0.3)',
                  borderRadius: '6px',
                  color: '#cbd5e1',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(51, 65, 85, 0.7)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(51, 65, 85, 0.5)';
                }}
              >
                Cancel
              </button>

              <button
                onClick={handleAnnotationSubmit}
                disabled={!annotationText.trim()}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: 500,
                  background: annotationText.trim() ? 'rgba(59, 130, 246, 0.6)' : 'rgba(51, 65, 85, 0.3)',
                  border: `1px solid ${annotationText.trim() ? 'rgba(59, 130, 246, 0.7)' : 'rgba(148, 163, 184, 0.2)'}`,
                  borderRadius: '6px',
                  color: annotationText.trim() ? '#f1f5f9' : '#64748b',
                  cursor: annotationText.trim() ? 'pointer' : 'not-allowed',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  if (annotationText.trim()) {
                    e.currentTarget.style.background = 'rgba(59, 130, 246, 0.8)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (annotationText.trim()) {
                    e.currentTarget.style.background = 'rgba(59, 130, 246, 0.6)';
                  }
                }}
              >
                Add Annotation
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Home Icon Button - Top Right */}
      <button
        onClick={() => window.location.hash = '/'}
        title="Back to Home"
        style={{
          position: "fixed",
          top: 12,
          right: 12,
          zIndex: 50,
          background: 'rgba(30, 41, 59, 0.5)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          border: '1px solid rgba(148, 163, 184, 0.15)',
          borderRadius: '6px',
          padding: '8px 12px',
          cursor: 'pointer',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '4px',
          transition: 'all 0.2s ease',
          boxShadow: '0 2px 8px 0 rgba(0, 0, 0, 0.15)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(30, 41, 59, 0.75)';
          e.currentTarget.style.transform = 'scale(1.05)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(30, 41, 59, 0.5)';
          e.currentTarget.style.transform = 'scale(1)';
        }}
      >
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#9cc3ff"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          <polyline points="9 22 9 12 15 12 15 22" />
        </svg>
        <span style={{ fontSize: '10px', color: '#9cc3ff', fontWeight: 500 }}>Home</span>
      </button>

      <div className="viewer-main">
        <div className="viewer-header">
          <div>Series: Axial Series 1 of 1</div>
          <div>
            {hasImage ? (() => {
              const selectedWindow = dynamicWindows.find(w => w.id === selectedWindowId);
              if (!selectedWindow) return '—';
              
              // Calculate max slices based on selected window's view
              let maxSliceForView = 0;
              if (selectedWindow.view === 'axial') maxSliceForView = dimensions[2] - 1;
              else if (selectedWindow.view === 'coronal') maxSliceForView = dimensions[1] - 1;
              else if (selectedWindow.view === 'sagittal') maxSliceForView = dimensions[0] - 1;
              else return '—'; // 3D view has no slices
              
              return `Image: ${selectedWindow.slice + 1} of ${maxSliceForView + 1}`;
            })() : 'Image: —'}
          </div>
        </div>

        <div className="viewer-screen" ref={wrapperRef} style={{ position: 'relative' }}>
          {/* Dynamic grid layout */}
          <div style={{
            display: 'grid',
            width: '100%',
            height: '100%',
            gap: '2px',
            gridTemplateColumns: dynamicWindows.length === 1 ? '1fr' :
                                 dynamicWindows.length === 2 ? '1fr 1fr' :
                                 dynamicWindows.length === 3 ? '1fr 1fr' :
                                 '1fr 1fr',
            gridTemplateRows: dynamicWindows.length === 1 ? '1fr' :
                             dynamicWindows.length === 2 ? '1fr' :
                             dynamicWindows.length === 3 ? '1fr 1fr' :
                             '1fr 1fr'
          }}>
            {dynamicWindows.map((window, idx) => (
              <div 
                key={window.id} 
                className="viewer-cell"
                onClick={() => setSelectedWindowId(window.id)}
                style={{
                  gridColumn: dynamicWindows.length === 3 && idx === 0 ? 'span 2' : 'auto',
                  border: selectedWindowId === window.id 
                    ? '3px solid rgba(59, 130, 246, 0.8)' 
                    : '1px solid #2a2f3d',
                  boxShadow: selectedWindowId === window.id
                    ? '0 0 20px rgba(59, 130, 246, 0.5)'
                    : 'none',
                  cursor: mode === 'measure' ? 'crosshair' :
                         mode === 'annotate' ? 'cell' :
                         mode === 'erase' ? 'not-allowed' :
                         interactionMode === 'pan' ? 'grab' :
                         'default', // Don't use 'pointer' - let measurement modes take precedence
                  transition: 'all 0.2s ease',
                }}
              >
                {/* Window controls */}
                <div style={{
                  position: 'absolute',
                  top: 8,
                  left: 8,
                  right: 8,
                  zIndex: 10,
                  display: 'flex',
                  gap: 8,
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}>
                  {/* View dropdown */}
                  <select
                    value={window.view}
                    onChange={(e) => changeWindowView(window.id, e.target.value as ViewType)}
                    style={{
                      background: 'rgba(30, 41, 59, 0.9)',
                      backdropFilter: 'blur(8px)',
                      WebkitBackdropFilter: 'blur(8px)',
                      color: '#e2e8f0',
                      border: '1px solid rgba(59, 130, 246, 0.3)',
                      borderRadius: 6,
                      padding: '6px 12px',
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: 'pointer',
                      outline: 'none',
                      transition: 'all 0.2s ease',
                      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.6)';
                      e.currentTarget.style.background = 'rgba(30, 41, 59, 0.95)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                      e.currentTarget.style.background = 'rgba(30, 41, 59, 0.9)';
                    }}
                  >
                    <option value="axial">Axial</option>
                    <option value="coronal">Coronal</option>
                    <option value="sagittal">Sagittal</option>
                    <option value="3d">3D Render</option>
                  </select>

                  {/* Delete button */}
                  {dynamicWindows.length > 1 && (
                    <button
                      onClick={() => removeWindow(window.id)}
                      title="Close window"
                      style={{
                        background: 'rgba(239, 68, 68, 0.15)',
                        backdropFilter: 'blur(8px)',
                        WebkitBackdropFilter: 'blur(8px)',
                        border: '1px solid rgba(239, 68, 68, 0.4)',
                        borderRadius: 6,
                        color: '#ef4444',
                        cursor: 'pointer',
                        padding: '4px 8px',
                        fontSize: 14,
                        fontWeight: 600,
                        lineHeight: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'all 0.2s ease',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(239, 68, 68, 0.3)';
                        e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.6)';
                        e.currentTarget.style.color = '#fca5a5';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)';
                        e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                        e.currentTarget.style.color = '#ef4444';
                      }}
                    >
                      ✕
                    </button>
                  )}
                </div>

                {/* Window content - separate canvases for 2D vs 3D to avoid context conflicts */}
         {!hasImage ? (
        <div className="empty-overlay">
          <div className="empty-box">
            <h3>No Image Loaded</h3>
            <p>Upload a .nii file to begin viewing</p>
          </div>
        </div>
                ) : window.view === '3d' ? (
          <canvas
                    key={`3d-${window.id}`} // Force remount when switching to 3D
                    ref={el => { 
                      if (el) {
                        canvasRefs.current[idx] = el;
                      }
                    }}
            style={{
              display: "block",
              width: "100%",
              height: "100%",
                      pointerEvents: 'auto', // Enable interaction for 3D
            }}
          />
      ) : (
        <>
          <canvas
                      key={`2d-${window.id}`} // Force remount when switching to 2D
                      ref={el => { 
                        if (el) {
                          canvasRefs.current[idx] = el;
                        }
                      }}
            style={{
              display: "block",
              width: "100%",
              height: "100%",
                        pointerEvents: 'none',
            }}
          />
                    {/* Overlay canvas for measurements and annotations */}
          <canvas
                      ref={el => {
                        if (el) {
                          overlayRefs.current[idx] = el;
                        }
                      }}
            style={{
                        position: 'absolute',
              top: 0,
              left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: mode !== 'none' ? 'auto' : 'none', // Enable clicks when measurement mode active
            }}
          />
        </>
      )}
    </div>
  ))}
</div>

          <span className="orientation top">A</span>
          <span className="orientation left">R</span>
          <span className="orientation right">L</span>
        </div>
      </div>

      <div className="viewer-sidebar" style={{ padding: 0, overflowY: 'auto', height: '100vh' }}>
          <div style={{ width: '100%' }}>
          <div className="sidebar-title" style={{
            padding: '24px 20px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: 'rgba(30, 41, 59, 0.3)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
            marginBottom: '0px',
          }}>
            <span style={{
              fontSize: '18px',
              fontWeight: 600,
              color: '#f1f5f9',
              letterSpacing: '0.5px',
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
            }}>DICOM Viewer</span>
          </div>

          {/* VIEW Section */}
          <div style={{ marginBottom: '0px', marginTop: '0px' }}>
              <button
              onClick={() => toggleSection('view')}
              style={{
                width: '100%',
                padding: '24px 16px',
                background: expandedSections.view
                  ? 'rgba(51, 65, 85, 0.6)'
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '16px',
                fontWeight: 600,
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'all 0.3s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                {/* Eye/Viewing Icon */}
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                  <circle cx="12" cy="12" r="3"/>
                </svg>
                <span>Viewing</span>
              </div>
              <span style={{ fontSize: '18px' }}>{expandedSections.view ? '▼' : '▶'}</span>
              </button>
            {expandedSections.view && (
              <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* Views Section - Removed individual view buttons since we have dropdowns on each window */}

                {/* Interaction Mode Icons */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '8px 0' }}>
                  {/* First Row: Scroll, Zoom, Pan */}
                  <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-start' }}>
                    {/* Scroll/Stack Icon */}
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ height: '14px', marginBottom: '2px' }} />
              <button
                        onClick={() => activateInteractionMode(interactionMode === 'scroll' ? null : 'scroll')}
                        title="Scroll through slices"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          padding: '8px',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          opacity: interactionMode === 'scroll' ? 1 : 0.4,
                          transform: interactionMode === 'scroll' ? 'scale(1.1)' : 'scale(1)',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={interactionMode === 'scroll' ? '#3b82f6' : '#94a3b8'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="1"/>
                          <circle cx="12" cy="5" r="1"/>
                          <circle cx="12" cy="19" r="1"/>
                          <path d="M12 2v3m0 14v3M12 8v8"/>
                          <polyline points="15 11 12 14 9 11"/>
                        </svg>
                        <span style={{ fontSize: '9px', color: interactionMode === 'scroll' ? '#3b82f6' : '#94a3b8', fontWeight: 500 }}>Scroll</span>
              </button>
                    </div>

                    {/* Zoom/Magnify Icon */}
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ height: '14px', marginBottom: '2px' }} />
              <button
                        onClick={() => activateInteractionMode(interactionMode === 'zoom' ? null : 'zoom')}
                        title="Zoom in/out"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          padding: '8px',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          opacity: interactionMode === 'zoom' ? 1 : 0.4,
                          transform: interactionMode === 'zoom' ? 'scale(1.1)' : 'scale(1)',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={interactionMode === 'zoom' ? '#3b82f6' : '#94a3b8'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="11" cy="11" r="8"/>
                          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                          <line x1="11" y1="8" x2="11" y2="14"/>
                          <line x1="8" y1="11" x2="14" y2="11"/>
                        </svg>
                        <span style={{ fontSize: '9px', color: interactionMode === 'zoom' ? '#3b82f6' : '#94a3b8', fontWeight: 500 }}>Zoom</span>
              </button>
                    </div>

                    {/* Pan/Hand Icon */}
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ height: '14px', marginBottom: '2px' }} />
              <button
                        onClick={() => activateInteractionMode(interactionMode === 'pan' ? null : 'pan')}
                        title="Pan image"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          padding: '8px',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          opacity: interactionMode === 'pan' ? 1 : 0.4,
                          transform: interactionMode === 'pan' ? 'scale(1.1)' : 'scale(1)',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={interactionMode === 'pan' ? '#3b82f6' : '#94a3b8'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/>
                          <path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/>
                          <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/>
                          <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>
                        </svg>
                        <span style={{ fontSize: '9px', color: interactionMode === 'pan' ? '#3b82f6' : '#94a3b8', fontWeight: 500 }}>Pan</span>
              </button>
            </div>
          </div>

                  {/* Second Row: Brightness, Contrast */}
                  <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-start' }}>
                    {/* Brightness/Level Icon */}
                <button
                      onClick={() => activateInteractionMode(interactionMode === 'brightness' ? null : 'brightness')}
                      title="Adjust brightness"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        padding: '8px',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        opacity: interactionMode === 'brightness' ? 1 : 0.4,
                        transform: interactionMode === 'brightness' ? 'scale(1.1)' : 'scale(1)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                    >
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={interactionMode === 'brightness' ? '#3b82f6' : '#94a3b8'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="5"/>
                        <line x1="12" y1="1" x2="12" y2="3"/>
                        <line x1="12" y1="21" x2="12" y2="23"/>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                        <line x1="1" y1="12" x2="3" y2="12"/>
                        <line x1="21" y1="12" x2="23" y2="12"/>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                      </svg>
                      <span style={{ fontSize: '9px', color: interactionMode === 'brightness' ? '#3b82f6' : '#94a3b8', fontWeight: 500 }}>Brightness</span>
                </button>

                    {/* Contrast/Width Icon */}
                <button
                      onClick={() => activateInteractionMode(interactionMode === 'contrast' ? null : 'contrast')}
                      title="Adjust contrast"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        padding: '8px',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        opacity: interactionMode === 'contrast' ? 1 : 0.4,
                        transform: interactionMode === 'contrast' ? 'scale(1.1)' : 'scale(1)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                    >
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={interactionMode === 'contrast' ? '#3b82f6' : '#94a3b8'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 2 A 10 10 0 0 1 12 22 Z" fill={interactionMode === 'contrast' ? '#3b82f6' : '#94a3b8'}/>
                      </svg>
                      <span style={{ fontSize: '9px', color: interactionMode === 'contrast' ? '#3b82f6' : '#94a3b8', fontWeight: 500 }}>Contrast</span>
                </button>
              </div>
                </div>

          </div>
          )}
          </div>

          {/* TOOLS Section */}
          <div style={{ marginBottom: '0px' }}>
            <button
              onClick={() => toggleSection('tools')}
              style={{
                width: '100%',
                padding: '24px 16px',
                background: expandedSections.tools
                  ? 'rgba(51, 65, 85, 0.6)'
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '16px',
                fontWeight: 600,
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'all 0.3s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                {/* Ruler/Measuring Icon */}
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 2h6v2h-6z"/>
                  <path d="M4 5h16v2H4z"/>
                  <path d="M7 9h10v2H7z"/>
                  <path d="M9 13h6v2H9z"/>
                  <path d="M11 17h2v5h-2z"/>
                  <rect x="2" y="2" width="20" height="20" rx="2"/>
                </svg>
                <span>Measuring</span>
              </div>
              <span style={{ fontSize: '18px' }}>{expandedSections.tools ? '▼' : '▶'}</span>
            </button>
            {expandedSections.tools && (
              <div style={{ padding: '16px', display: 'flex', justifyContent: 'center', gap: '30px' }}>
                {/* Measure Tool */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                  <button
                    onClick={() => activateMeasuringMode(mode === 'measure' ? 'none' : 'measure')}
                    title="Measure distances"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      padding: '0',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: mode === 'measure' ? 1 : 0.6,
                      transform: mode === 'measure' ? 'scale(1.1)' : 'scale(1)',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.opacity = '1';
                      e.currentTarget.style.transform = 'scale(1.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.opacity = mode === 'measure' ? '1' : '0.6';
                      e.currentTarget.style.transform = mode === 'measure' ? 'scale(1.1)' : 'scale(1)';
                    }}
                  >
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={mode === 'measure' ? '#60a5fa' : '#94a3b8'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.4 2.4 0 0 1 0-3.4l2.6-2.6a2.4 2.4 0 0 1 3.4 0Z"/>
                      <path d="m14.5 12.5 2-2"/>
                      <path d="m11.5 9.5 2-2"/>
                      <path d="m8.5 6.5 2-2"/>
                      <path d="m17.5 15.5 2-2"/>
                    </svg>
                  </button>
                  <span style={{ fontSize: '11px', color: mode === 'measure' ? '#60a5fa' : '#94a3b8', fontWeight: 500 }}>Measure</span>
                </div>

                {/* Annotate Tool */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                  <button
                    onClick={() => activateMeasuringMode(mode === 'annotate' ? 'none' : 'annotate')}
                    title="Add annotations"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      padding: '0',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: mode === 'annotate' ? 1 : 0.6,
                      transform: mode === 'annotate' ? 'scale(1.1)' : 'scale(1)',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.opacity = '1';
                      e.currentTarget.style.transform = 'scale(1.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.opacity = mode === 'annotate' ? '1' : '0.6';
                      e.currentTarget.style.transform = mode === 'annotate' ? 'scale(1.1)' : 'scale(1)';
                    }}
                  >
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={mode === 'annotate' ? '#60a5fa' : '#94a3b8'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                    </svg>
                  </button>
                  <span style={{ fontSize: '11px', color: mode === 'annotate' ? '#60a5fa' : '#94a3b8', fontWeight: 500 }}>Annotate</span>
                </div>

                {/* Erase Tool */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                  <button
                    onClick={() => activateMeasuringMode(mode === 'erase' ? 'none' : 'erase')}
                    title="Erase measurements and annotations"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      padding: '0',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: mode === 'erase' ? 1 : 0.6,
                      transform: mode === 'erase' ? 'scale(1.1)' : 'scale(1)',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.opacity = '1';
                      e.currentTarget.style.transform = 'scale(1.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.opacity = mode === 'erase' ? '1' : '0.6';
                      e.currentTarget.style.transform = mode === 'erase' ? 'scale(1.1)' : 'scale(1)';
                    }}
                  >
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={mode === 'erase' ? '#f87171' : '#94a3b8'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/>
                      <path d="M22 21H7"/>
                      <path d="m5 11 9 9"/>
                    </svg>
                  </button>
                  <span style={{ fontSize: '11px', color: mode === 'erase' ? '#f87171' : '#94a3b8', fontWeight: 500 }}>Erase</span>
                </div>
              </div>
            )}
          </div>

          {/* WINDOWS Section */}
          <div style={{ marginBottom: '0px' }}>
                <button
              onClick={() => toggleSection('windows')}
              style={{
                width: '100%',
                padding: '24px 16px',
                background: expandedSections.windows
                  ? 'rgba(51, 65, 85, 0.6)'
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '16px',
                fontWeight: 600,
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'all 0.3s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                {/* Grid Icon */}
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7"/>
                  <rect x="14" y="3" width="7" height="7"/>
                  <rect x="14" y="14" width="7" height="7"/>
                  <rect x="3" y="14" width="7" height="7"/>
                </svg>
                <span>Windows</span>
              </div>
              <span style={{ fontSize: '18px' }}>{expandedSections.windows ? '▼' : '▶'}</span>
            </button>
            {expandedSections.windows && (
              <div style={{ padding: '16px' }}>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{
                    fontSize: '13px',
                    fontWeight: 600,
                    color: '#94a3b8',
                    marginBottom: '12px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px'
                  }}>
                    Manage Windows
                  </div>

                  <button
                    className="sidebar-btn"
                    onClick={addWindow}
                    disabled={dynamicWindows.length >= 4}
                    style={{
                        width: '100%',
                        marginBottom: '8px',
                        opacity: dynamicWindows.length >= 4 ? 0.5 : 1,
                        cursor: dynamicWindows.length >= 4 ?'not-allowed' : 'pointer',
                    }}
                  >
                    + Add Window ({dynamicWindows.length}/4)
                  </button>

                  <div style={{ fontSize: '12px', color: '#64748b', marginTop: '12px', marginBottom: '4px' }}>
                    Active windows: {dynamicWindows.length}
                </div>

                  <div style={{
                    fontSize: '12px', 
                    color: '#3b82f6', 
                    marginTop: '4px',
                    padding: '8px',
                    background: 'rgba(59, 130, 246, 0.1)',
                    borderRadius: '4px',
                    border: '1px solid rgba(59, 130, 246, 0.3)'
                  }}>
                    Selected: Window {selectedWindowId}
                    {dynamicWindows.find(w => w.id === selectedWindowId) && 
                      ` (${dynamicWindows.find(w => w.id === selectedWindowId)!.view.charAt(0).toUpperCase() + 
                        dynamicWindows.find(w => w.id === selectedWindowId)!.view.slice(1)})`
                    }
                  </div>
                </div>
              </div>
            )}
          </div>


          {/* UPLOAD Section */}
          <div style={{ marginBottom: '0px' }}>
                <button
              onClick={() => toggleSection('upload')}
              style={{
                width: '100%',
                padding: '24px 16px',
                background: expandedSections.upload
                  ? 'rgba(51, 65, 85, 0.6)'
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '16px',
                fontWeight: 600,
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'all 0.3s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                {/* Folder/Upload Icon */}
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                  <polyline points="12 11 12 17"/>
                  <polyline points="9 14 12 11 15 14"/>
                </svg>
                <span>Data</span>
              </div>
              <span style={{ fontSize: '18px' }}>{expandedSections.upload ? '▼' : '▶'}</span>
                </button>
            {expandedSections.upload && (
  <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
    <input id="niftiUpload" type="file" accept=".nii,.nii.gz" multiple className="upload-box" />

    {/* Model Selector */}
    <div style={{
      background: 'rgba(51, 65, 85, 0.4)',
      borderRadius: '8px',
      padding: '12px',
      border: '1px solid rgba(148, 163, 184, 0.2)',
    }}>
      <label style={{
        color: 'rgba(255, 255, 255, 0.8)',
        fontSize: '13px',
        fontWeight: 600,
        marginBottom: '8px',
        display: 'block',
      }}>
        Segmentation Model
      </label>
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={() => setSelectedModel('unet')}
          disabled={isPredicting}
          style={{
            flex: 1,
            padding: '10px',
            background: selectedModel === 'unet' ? 'rgba(59, 130, 246, 0.6)' : 'rgba(51, 65, 85, 0.3)',
            border: selectedModel === 'unet' ? '2px solid rgba(59, 130, 246, 0.8)' : '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '6px',
            color: '#fff',
            fontSize: '14px',
            fontWeight: selectedModel === 'unet' ? 600 : 500,
            cursor: isPredicting ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
          }}
        >
          U-Net
        </button>
        <button
          onClick={() => setSelectedModel('pspnet')}
          disabled={isPredicting}
          style={{
            flex: 1,
            padding: '10px',
            background: selectedModel === 'pspnet' ? 'rgba(59, 130, 246, 0.6)' : 'rgba(51, 65, 85, 0.3)',
            border: selectedModel === 'pspnet' ? '2px solid rgba(59, 130, 246, 0.8)' : '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '6px',
            color: '#fff',
            fontSize: '14px',
            fontWeight: selectedModel === 'pspnet' ? 600 : 500,
            cursor: isPredicting ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
          }}
        >
          PSPNet
        </button>
      </div>
    </div>

    {/* Predict Button */}
    <button
      onClick={handlePrediction}
      disabled={!niftiData || !mriId || isPredicting}
      style={{
        background: isPredicting 
          ? 'rgba(100, 116, 139, 0.4)' 
          : 'rgba(59,130,246,0.6)',
        border: `1px solid ${isPredicting ? 'rgba(100, 116, 139, 0.5)' : 'rgba(59,130,246,0.7)'}`,
        borderRadius: '6px',
        color: '#fff',
        padding: '12px',
        fontWeight: 600,
        cursor: (!niftiData || !mriId || isPredicting) ? 'not-allowed' : 'pointer',
        opacity: (!niftiData || !mriId || isPredicting) ? 0.5 : 1,
        transition: 'all 0.2s ease',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
      }}
      title={!mriId ? "Backend not available - prediction features require backend server" : ""}
    >
      {isPredicting ? (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '16px',
              height: '16px',
              border: '3px solid rgba(255,255,255,0.3)',
              borderTop: '3px solid #fff',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }} />
            Running...
          </div>
          <div style={{ fontSize: '11px', opacity: 0.8, textAlign: 'center' }}>
            {predictionProgress}
          </div>
        </>
      ) : (
        'Run Prediction'
      )}
    </button>
    
    <style>{`
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `}</style>

    {/* Overlay Toggle */}
    <button
      onClick={() => setShowOverlay(o => !o)}
      disabled={!predictionMask}
      style={{
        background: showOverlay ? 'rgba(34,197,94,0.6)' : 'rgba(51,65,85,0.6)',
        border: showOverlay ? '1px solid rgba(34,197,94,0.8)' : '1px solid rgba(148,163,184,0.3)',
        borderRadius: '6px',
        color: '#fff',
        padding: '10px',
        fontWeight: 600,
        cursor: predictionMask ? 'pointer' : 'not-allowed',
        opacity: predictionMask ? 1 : 0.5,
        transition: 'all 0.2s ease',
      }}
    >
      {showOverlay ? 'Hide Overlay' : 'Show Overlay'}
    </button>

    {/* Download Segmentation Button */}
    <button
      onClick={downloadSegmentationAsNifti}
      disabled={!predictionMask}
      style={{
        background: predictionMask ? 'rgba(59,130,246,0.6)' : 'rgba(51,65,85,0.6)',
        border: predictionMask ? '1px solid rgba(59,130,246,0.6)' : '1px solid rgba(148,163,184,0.3)',
        borderRadius: '6px',
        color: '#fff',
        padding: '10px',
        fontWeight: 600,
        cursor: predictionMask ? 'pointer' : 'not-allowed',
        opacity: predictionMask ? 1 : 0.5,
        transition: 'all 0.2s ease',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
      }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Download Segmentation
    </button>
        </div>
)}

        </div>
      </div>
      </div>

      {/* Reset Button - Bottom Right */}
      <button
        onClick={resetView}
        title="Reset view to default"
        style={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          zIndex: 50,
          background: 'rgba(59, 130, 246, 0.4)',
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          border: '1px solid rgba(59, 130, 246, 0.5)',
          borderRadius: '8px',
          padding: '10px 16px',
          cursor: 'pointer',
          color: '#fff',
          fontSize: '13px',
          fontWeight: 600,
          transition: 'all 0.2s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          boxShadow: '0 4px 16px 0 rgba(59, 130, 246, 0.3)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(59, 130, 246, 0.6)';
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 6px 20px 0 rgba(59, 130, 246, 0.4)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(59, 130, 246, 0.4)';
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = '0 4px 16px 0 rgba(59, 130, 246, 0.3)';
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
          <path d="M21 3v5h-5"/>
          <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
          <path d="M3 21v-5h5"/>
        </svg>
        Reset View
      </button>

      {viewerWindows.map(window =>(
        <DraggableViewerWindow
          key={window.id}
          window={window}
          onClose={() => closeViewerWindow(window.id)}
          onMinimize={() => toggleMinimizeWindow(window.id)}
          onLoadFile={(file) => loadFileIntoWindow(window.id, file)}
          onLoadCurrent={() => loadCurrentFileIntoWindow(window.id)}
          onPositionChange={(x, y) => updateWindowPosition(window.id, x, y)}
        />
      ))}

    </div>
  );
};

export default DicomViewer;