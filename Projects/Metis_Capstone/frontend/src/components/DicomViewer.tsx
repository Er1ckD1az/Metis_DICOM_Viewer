import React, { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import "./DicomViewer.css";
import * as nifti from "nifti-reader-js";

const DicomViewer: React.FC = () => {
  const location = useLocation();
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const [hasImage, setHasImage] = useState(false);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [maxSlices, setMaxSlices] = useState(0);
  const [currentView, setCurrentView] = useState<'axial' | 'coronal' | 'sagittal'>('axial');
  const [niftiData, setNiftiData] = useState<Float32Array | null>(null);
  const [dimensions, setDimensions] = useState<[number, number, number]>([0, 0, 0]);
  const [windowLevel, setWindowLevel] = useState(200);
  const [windowWidth, setWindowWidth] = useState(600);

  // overlay tools state
  const [mode, setMode] = useState<'select'|'measure'|'annotate'|'crosshair'>('select');
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const drawingRef = useRef<{ drawing: boolean; x:number; y:number } | null>(null);
  const [measurements, setMeasurements] = useState<Array<{ x1:number,y1:number,x2:number,y2:number }>>([]);
  const [annotations, setAnnotations] = useState<Array<{ x:number,y:number,text:string }>>([]);

  // Load and display NIfTI file (same logic as original)
  const loadNiftiFile = async (file: File) => {
    try {
      console.log("Loading NIfTI file:", file.name);
      const arrayBuffer = await file.arrayBuffer();
      if (!nifti.isNIFTI(arrayBuffer)) throw new Error("Not a valid NIfTI file");
      const header = nifti.readHeader(arrayBuffer);
      const dims = [header.dims[1], header.dims[2], header.dims[3]];
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
      setMaxSlices(z - 1);
      setCurrentSlice(Math.floor(z/2));
      setHasImage(true);
      console.log("NIfTI loaded");
    } catch (error) {
      console.error("Failed to load NIfTI file:", error);
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

  // core render - closely mirrors original behavior
  const renderSlice = () => {
    if (!niftiData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const overlay = overlayRef.current;

    const { slice, width, height } = getSlice(niftiData, dimensions, currentSlice, currentView);

    // target viewer size
    const viewerWidth = 800;
    const viewerHeight = 600;
    const scale = Math.max(viewerWidth / width, viewerHeight / height);
    const displayWidth = Math.floor(width * scale);
    const displayHeight = Math.floor(height * scale);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;

    // scale context for DPR
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // window/level
    const wlMin = windowLevel - windowWidth / 2;
    const wlMax = windowLevel + windowWidth / 2;
    const windowed = new Uint8ClampedArray(width * height);

    for (let j=0;j<height;j++){
      for (let i=0;i<width;i++){
        const val = slice[i + j*width];
        if (val < wlMin) windowed[i + j*width] = 0;
        else if (val > wlMax) windowed[i + j*width] = 255;
        else {
          const normalized = (val - wlMin) / (wlMax - wlMin);
          windowed[i + j*width] = Math.floor(Math.pow(normalized, 0.9) * 255);
        }
      }
    }

    const imageData = ctx.createImageData(displayWidth, displayHeight);
    const scaleInv = 1 / scale;

    for (let yPix=0;yPix<displayHeight;yPix++){
      for (let xPix=0;xPix<displayWidth;xPix++){
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

        const pixelIndex = (yPix * displayWidth + xPix) * 4;
        imageData.data[pixelIndex] = val;
        imageData.data[pixelIndex + 1] = val;
        imageData.data[pixelIndex + 2] = val;
        imageData.data[pixelIndex + 3] = 255;
      }
    }

    // apply flip transforms to canvas via CSS transform (keeps drawing coordinates consistent)
    const flipX = flipH ? -1 : 1;
    const flipY = flipV ? -1 : 1;
    (canvas.style as any).transform = `scale(${flipX}, ${flipY})`;

    ctx.setTransform(1,0,0,1,0,0);
    ctx.putImageData(imageData, 0, 0);

    // overlay sizing and draw
    if (overlay) {
      overlay.width = displayWidth * dpr;
      overlay.height = displayHeight * dpr;
      overlay.style.width = `${displayWidth}px`;
      overlay.style.height = `${displayHeight}px`;
      const octx = overlay.getContext('2d');
      if (octx) {
        octx.setTransform(dpr,0,0,dpr,0,0);
        octx.clearRect(0,0,displayWidth,displayHeight);

        // draw measurements
        octx.strokeStyle = '#00FF00';
        octx.lineWidth = 2;
        octx.fillStyle = '#00FF00';
        measurements.forEach(m => {
          octx.beginPath();
          octx.moveTo(m.x1, m.y1);
          octx.lineTo(m.x2, m.y2);
          octx.stroke();
          const dx = m.x2 - m.x1;
          const dy = m.y2 - m.y1;
          const dist = Math.sqrt(dx*dx + dy*dy).toFixed(1);
          octx.fillText(`${dist}px`, (m.x1 + m.x2)/2 + 5, (m.y1 + m.y2)/2 - 5);
        });

        // draw annotations
        octx.fillStyle = '#FFFF00';
        annotations.forEach(a => {
          octx.beginPath();
          octx.arc(a.x, a.y, 4, 0, Math.PI*2);
          octx.fill();
          octx.fillText(a.text, a.x + 6, a.y - 6);
        });

        // crosshair mode center
        if (mode === 'crosshair'){
          const cx = displayWidth / 2;
          const cy = displayHeight / 2;
          octx.strokeStyle = '#FF0000';
          octx.beginPath();
          octx.moveTo(cx, 0);
          octx.lineTo(cx, displayHeight);
          octx.moveTo(0, cy);
          octx.lineTo(displayWidth, cy);
          octx.stroke();
        }
      }
    }
  };

  useEffect(() => {
    renderSlice();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [niftiData, currentSlice, currentView, windowLevel, windowWidth, flipH, flipV, measurements, annotations, mode]);

  // file input handler
  useEffect(() => {
    const input = document.getElementById('niftiUpload') as HTMLInputElement | null;
    if (!input) return;
    const handleFileChange = async (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      if (file.name.toLowerCase().endsWith('.nii') || file.name.toLowerCase().endsWith('.nii.gz')) {
        await loadNiftiFile(file);
      } else {
        console.error('Please upload a .nii or .nii.gz file');
      }
    };
    input.addEventListener('change', handleFileChange);
    return () => input.removeEventListener('change', handleFileChange);
  }, []);

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
    
    // Handle demo mode - load sample file
    else if (state?.demoMode) {
      console.log("Demo mode activated - loading sample file");
      loadDemoFile();
    }
  }, [location]);

  // Load demo NIfTI file from public folder
  const loadDemoFile = async () => {
    try {
      const response = await fetch('/demo-sample.nii');
      if (!response.ok) {
        throw new Error('Demo file not found. Please add demo-sample.nii to the public folder.');
      }
      const arrayBuffer = await response.arrayBuffer();
      const blob = new Blob([arrayBuffer]);
      const file = new File([blob], 'demo-sample.nii', { type: 'application/octet-stream' });
      await loadNiftiFile(file);
      console.log("Demo file loaded successfully");
    } catch (error) {
      console.error("Failed to load demo file:", error);
      alert("Demo file not found. Please add a demo-sample.nii file to the public folder.");
    }
  };

  // overlay events for measure/annotate
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const rect = () => overlay.getBoundingClientRect();

    const toLocal = (ev: MouseEvent) => {
      const r = rect();
      return { x: ev.clientX - r.left, y: ev.clientY - r.top };
    };

    const onDown = (ev: MouseEvent) => {
      if (mode === 'measure' || mode === 'annotate'){
        const p = toLocal(ev);
        drawingRef.current = { drawing: true, x: p.x, y: p.y };
      }
    };

    const onMove = (ev: MouseEvent) => {
      if (!drawingRef.current) return;
      const p = toLocal(ev);
      if (mode === 'measure'){
        // live preview: redraw overlays by calling renderSlice (which clears overlays) then draw line
        renderSlice();
        const octx = overlay.getContext('2d');
        if (!octx) return;
        const dpr = window.devicePixelRatio || 1;
        octx.setTransform(dpr,0,0,dpr,0,0);
        octx.beginPath();
        octx.moveTo(drawingRef.current.x, drawingRef.current.y);
        octx.lineTo(p.x, p.y);
        octx.strokeStyle = '#00FF00';
        octx.lineWidth = 2;
        octx.stroke();
        const dx = p.x - drawingRef.current.x;
        const dy = p.y - drawingRef.current.y;
        octx.fillStyle = '#00FF00';
        octx.fillText(`${Math.sqrt(dx*dx + dy*dy).toFixed(1)}px`, (drawingRef.current.x + p.x)/2 + 5, (drawingRef.current.y + p.y)/2 - 5);
      } else if (mode === 'annotate'){
        renderSlice();
        const octx = overlay.getContext('2d');
        if (!octx) return;
        const dpr = window.devicePixelRatio || 1;
        octx.setTransform(dpr,0,0,dpr,0,0);
        octx.beginPath();
        octx.arc(p.x, p.y, 4, 0, Math.PI*2);
        octx.fillStyle = '#FFFF00';
        octx.fill();
      }
    };

    const onUp = (ev: MouseEvent) => {
      const ref = drawingRef.current;
      if (!ref) return;
      const p = toLocal(ev);

      if (mode === 'measure') {
        setMeasurements(ms => [
          ...ms,
          { x1: ref.x, y1: ref.y, x2: p.x, y2: p.y },
        ]);
      } else if (mode === 'annotate') {
        const text = prompt('Annotation text:', 'note') || 'note';
        setAnnotations(a => [...a, { x: p.x, y: p.y, text }]);
      }

      drawingRef.current = null;
      renderSlice();
    };

    overlay.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    return () => {
      overlay.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [mode, measurements, annotations]);

  const switchView = (view: 'axial'|'coronal'|'sagittal') => {
    setCurrentView(view);
    if (view === 'axial') setMaxSlices(dimensions[2] - 1);
    else if (view === 'coronal') setMaxSlices(dimensions[1] - 1);
    else setMaxSlices(dimensions[0] - 1);
    setCurrentSlice(Math.floor((view === 'axial' ? dimensions[2] : view === 'coronal' ? dimensions[1] : dimensions[0]) / 2));
  };

  const changeSlice = (direction: 'prev'|'next') => {
    if (direction === 'prev' && currentSlice > 0) setCurrentSlice(s => s - 1);
    else if (direction === 'next' && currentSlice < maxSlices) setCurrentSlice(s => s + 1);
  };

  const resetView = () => {
    setFlipH(false);
    setFlipV(false);
    setMeasurements([]);
    setAnnotations([]);
    setMode('select');
  };

  return (
    <div className="viewer-container">
      <div style={{ position: "fixed", top: 10, left: 10, zIndex: 50 }}>
        <a href="#/" style={{ color: "#9cc3ff", textDecoration: "none" }}>← Back to Home</a>
      </div>

      <div className="viewer-main">
        <div className="viewer-header">
          <div>Series: Axial Series 1 of 1</div>
          <div>Image: {hasImage ? `${currentSlice + 1} of ${maxSlices + 1}` : '—'}</div>
        </div>

        <div className="viewer-screen" ref={wrapperRef} style={{ position: 'relative' }}>
          {!hasImage && (
            <div className="empty-overlay">
              <div className="empty-box">
                <h3>No Image Loaded</h3>
                <p>Upload a .nii file to begin viewing</p>
              </div>
            </div>
          )}

          {hasImage && (
            <>
              <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', display: 'block' }} />
              <canvas ref={overlayRef} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'auto' }} />
            </>
          )}

          <span className="orientation top">A</span>
          <span className="orientation left">R</span>
          <span className="orientation right">L</span>
        </div>
      </div>

      <div className="viewer-sidebar">
        <div>
          <div className="sidebar-title">NIfTI Viewer</div>

          <div className="sidebar-section">
            <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
              VIEW
            </p>
            <div className="sidebar-buttons">
              <button className={`sidebar-btn ${currentView === 'axial' ? 'active' : ''}`} onClick={() => switchView('axial')}>Axial</button>
              <button className={`sidebar-btn ${currentView === 'coronal' ? 'active' : ''}`} onClick={() => switchView('coronal')}>Coronal</button>
              <button className={`sidebar-btn ${currentView === 'sagittal' ? 'active' : ''}`} onClick={() => switchView('sagittal')}>Sagittal</button>
            </div>
          </div>

          {hasImage && (
            <div className="sidebar-section">
              <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
                SLICE NAVIGATION
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                <button className="sidebar-btn" onClick={() => changeSlice('prev')} disabled={currentSlice === 0}>←</button>
                <span style={{ fontSize: '12px', color: '#ccc' }}>{currentSlice + 1} / {maxSlices + 1}</span>
                <button className="sidebar-btn" onClick={() => changeSlice('next')} disabled={currentSlice === maxSlices}>→</button>
              </div>
              <input type="range" min="0" max={maxSlices} value={currentSlice} onChange={(e) => setCurrentSlice(parseInt(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}

          {hasImage && (
            <div className="sidebar-section">
              <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
                WINDOW/LEVEL
              </p>
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '10px', color: '#ccc' }}>Level:</label>
                <input type="range" min="-1000" max="1000" value={windowLevel} onChange={(e) => setWindowLevel(parseInt(e.target.value))} style={{ width: '100%' }} />
                <span style={{ fontSize: '10px', color: '#ccc' }}>{windowLevel}</span>
              </div>
              <div>
                <label style={{ fontSize: '10px', color: '#ccc' }}>Width:</label>
                <input type="range" min="100" max="2000" value={windowWidth} onChange={(e) => setWindowWidth(parseInt(e.target.value))} style={{ width: '100%' }} />
                <span style={{ fontSize: '10px', color: '#ccc' }}>{windowWidth}</span>
              </div>
            </div>
          )}

          <div className="sidebar-section">
            <div className="sidebar-buttons">
              <button className={`sidebar-btn ${mode === 'select' ? 'active' : ''}`} onClick={() => setMode('select')}>Select</button>
              <button className={`sidebar-btn ${mode === 'measure' ? 'active' : ''}`} onClick={() => setMode('measure')}>Measure</button>
              <button className={`sidebar-btn ${mode === 'annotate' ? 'active' : ''}`} onClick={() => setMode('annotate')}>Annotate</button>
              <button className={`sidebar-btn ${mode === 'crosshair' ? 'active' : ''}`} onClick={() => setMode('crosshair')}>Crosshair</button>
            </div>
          </div>

          <div className="sidebar-section" style={{ marginTop: "15px" }}>
            <div className="sidebar-buttons">
              <button className="sidebar-btn" onClick={() => setFlipH(f => !f)}>Flip H</button>
              <button className="sidebar-btn" onClick={() => setFlipV(f => !f)}>Flip V</button>
              <button className="sidebar-btn" onClick={() => resetView()}>Reset</button>
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-buttons">
              <input id="niftiUpload" type="file" accept=".nii,.nii.gz" className="upload-box" />
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};

export default DicomViewer;

