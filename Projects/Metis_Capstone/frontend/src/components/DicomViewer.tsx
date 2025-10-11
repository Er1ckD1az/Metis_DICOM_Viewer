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
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const lastPanPos = useRef({ x: 0, y: 0 });

  // overlay tools state
  const [mode, setMode] = useState<'select'|'measure'|'annotate'|'crosshair'>('select');
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const drawingRef = useRef<{ drawing: boolean; x:number; y:number } | null>(null);
  const [measurements, setMeasurements] = useState<Array<{ x1:number,y1:number,x2:number,y2:number }>>([]);
  const [annotations, setAnnotations] = useState<Array<{ x:number,y:number,text:string }>>([]);
  
  // Accordion state for collapsible sections
  const [expandedSections, setExpandedSections] = useState<{[key: string]: boolean}>({
    view: true,
    tools: false,
    transform: false,
    upload: false,
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  // Interaction mode state (only one can be active at a time)
  const [interactionMode, setInteractionMode] = useState<'scroll' | 'zoom' | 'pan' | 'brightness' | 'contrast' | null>('scroll');

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
        
        // Threshold: if original value is very low, it's background
        // Set to viewer background color instead of showing it
        const backgroundThreshold = wlMin + (wlMax - wlMin) * 0.05; // 5% of window range
        if (origVal < backgroundThreshold) {
          // Set to viewer background color #2e2e2e = rgb(46, 46, 46)
          imageData.data[pixelIndex] = 46;
          imageData.data[pixelIndex + 1] = 46;
          imageData.data[pixelIndex + 2] = 46;
          imageData.data[pixelIndex + 3] = 255;
        } else {
          imageData.data[pixelIndex] = val;
          imageData.data[pixelIndex + 1] = val;
          imageData.data[pixelIndex + 2] = val;
          imageData.data[pixelIndex + 3] = 255;
        }
      }
    }

    // apply flip, zoom, and pan transforms to canvas via CSS transform
    const flipX = flipH ? -1 : 1;
    const flipY = flipV ? -1 : 1;
    (canvas.style as any).transform = `translate(${panOffset.x}px, ${panOffset.y}px) scale(${flipX * zoomLevel}, ${flipY * zoomLevel})`;

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
  }, [niftiData, currentSlice, currentView, windowLevel, windowWidth, flipH, flipV, measurements, annotations, mode, zoomLevel, panOffset]);

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

  // Interaction mode handlers (scroll, zoom, pan)
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || !hasImage) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      
      if (interactionMode === 'scroll') {
        // Scroll through slices
        if (e.deltaY < 0 && currentSlice > 0) {
          setCurrentSlice(s => s - 1);
        } else if (e.deltaY > 0 && currentSlice < maxSlices) {
          setCurrentSlice(s => s + 1);
        }
      } else if (interactionMode === 'zoom') {
        // Zoom in/out
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        setZoomLevel(z => Math.max(0.5, Math.min(5, z + delta)));
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
        if (interactionMode === 'pan') {
          const dx = e.clientX - lastPanPos.current.x;
          const dy = e.clientY - lastPanPos.current.y;
          setPanOffset(offset => ({ x: offset.x + dx, y: offset.y + dy }));
          lastPanPos.current = { x: e.clientX, y: e.clientY };
        } else if (interactionMode === 'brightness') {
          const dy = e.clientY - lastPanPos.current.y;
          setWindowLevel(level => Math.max(-1000, Math.min(1000, level - dy * 2)));
          lastPanPos.current = { x: e.clientX, y: e.clientY };
        } else if (interactionMode === 'contrast') {
          const dy = e.clientY - lastPanPos.current.y;
          setWindowWidth(width => Math.max(100, Math.min(2000, width - dy * 2)));
          lastPanPos.current = { x: e.clientX, y: e.clientY };
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
  }, [interactionMode, hasImage, currentSlice, maxSlices]);

  const switchView = (view: 'axial'|'coronal'|'sagittal') => {
    setCurrentView(view);
    if (view === 'axial') setMaxSlices(dimensions[2] - 1);
    else if (view === 'coronal') setMaxSlices(dimensions[1] - 1);
    else setMaxSlices(dimensions[0] - 1);
    setCurrentSlice(Math.floor((view === 'axial' ? dimensions[2] : view === 'coronal' ? dimensions[1] : dimensions[0]) / 2));
  };

  const resetView = () => {
    setFlipH(false);
    setFlipV(false);
    setMeasurements([]);
    setAnnotations([]);
    setMode('select');
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
    setInteractionMode('scroll');
    setWindowLevel(224);
    setWindowWidth(600);
  };

  return (
    <div className="viewer-container">
      <div style={{ 
        position: "fixed", 
        top: 10, 
        left: 10, 
        zIndex: 50,
        background: 'rgba(30, 41, 59, 0.6)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        border: '1px solid rgba(148, 163, 184, 0.2)',
        borderRadius: '8px',
        padding: '8px 16px',
        boxShadow: '0 4px 16px 0 rgba(0, 0, 0, 0.2)',
      }}>
        <a href="#/" style={{ 
          color: "#9cc3ff", 
          textDecoration: "none",
          fontSize: '14px',
          fontWeight: 500,
        }}>← Back to Home</a>
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

      <div className="viewer-sidebar" style={{ padding: 0 }}>
        <div style={{ width: '100%' }}>
          <div className="sidebar-title" style={{ padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>NIfTI Viewer</span>
              <button
              onClick={resetView}
              title="Reset view to default"
              style={{
                background: 'rgba(59, 130, 246, 0.4)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: '1px solid rgba(59, 130, 246, 0.5)',
                borderRadius: '6px',
                padding: '6px 10px',
                cursor: 'pointer',
                color: '#fff',
                fontSize: '11px',
                fontWeight: 600,
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(59, 130, 246, 0.6)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(59, 130, 246, 0.4)';
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
                <path d="M21 3v5h-5"/>
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
                <path d="M3 21v-5h5"/>
              </svg>
              Reset
              </button>
          </div>

          {/* VIEW Section */}
          <div style={{ marginBottom: '0px' }}>
              <button
              onClick={() => toggleSection('view')}
              style={{
                width: '100%',
                padding: '20px 16px',
                background: expandedSections.view 
                  ? 'rgba(51, 65, 85, 0.6)' 
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '15px',
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
              <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* View Buttons */}
            <div className="sidebar-buttons">
                  <button className={`sidebar-btn ${currentView === 'axial' ? 'active' : ''}`} onClick={() => switchView('axial')}>Axial</button>
                  <button className={`sidebar-btn ${currentView === 'coronal' ? 'active' : ''}`} onClick={() => switchView('coronal')}>Coronal</button>
                  <button className={`sidebar-btn ${currentView === 'sagittal' ? 'active' : ''}`} onClick={() => switchView('sagittal')}>Sagittal</button>
          </div>

                {/* Interaction Mode Icons */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '8px 0' }}>
                  {/* First Row: Scroll, Zoom, Pan */}
                  <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-start' }}>
                    {/* Scroll/Stack Icon with slice counter */}
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
                      {/* Tiny Slice Counter with +/- buttons */}
                      {hasImage && (
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '4px',
                          height: '14px',
                          marginBottom: '2px',
                        }}>
                          <button
                            onClick={() => setCurrentSlice(Math.max(0, currentSlice - 1))}
                            disabled={currentSlice === 0}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: currentSlice === 0 ? '#555' : '#94a3b8',
                              cursor: currentSlice === 0 ? 'not-allowed' : 'pointer',
                              fontSize: '10px',
                              padding: '0',
                              transition: 'color 0.2s ease',
                            }}
                            onMouseEnter={(e) => {
                              if (currentSlice !== 0) e.currentTarget.style.color = '#3b82f6';
                            }}
                            onMouseLeave={(e) => {
                              if (currentSlice !== 0) e.currentTarget.style.color = '#94a3b8';
                            }}
                          >
                            −
                          </button>
                          <span style={{ 
                            fontSize: '10px', 
                            color: '#94a3b8',
                          }}>
                            {currentSlice + 1}/{maxSlices + 1}
                          </span>
                          <button
                            onClick={() => setCurrentSlice(Math.min(maxSlices, currentSlice + 1))}
                            disabled={currentSlice === maxSlices}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: currentSlice === maxSlices ? '#555' : '#94a3b8',
                              cursor: currentSlice === maxSlices ? 'not-allowed' : 'pointer',
                              fontSize: '10px',
                              padding: '0',
                              transition: 'color 0.2s ease',
                            }}
                            onMouseEnter={(e) => {
                              if (currentSlice !== maxSlices) e.currentTarget.style.color = '#3b82f6';
                            }}
                            onMouseLeave={(e) => {
                              if (currentSlice !== maxSlices) e.currentTarget.style.color = '#94a3b8';
                            }}
                          >
                            +
                          </button>
                        </div>
                      )}
                      {!hasImage && <div style={{ height: '14px', marginBottom: '2px' }} />}

              <button
                        onClick={() => setInteractionMode(interactionMode === 'scroll' ? null : 'scroll')}
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
                        onClick={() => setInteractionMode(interactionMode === 'zoom' ? null : 'zoom')}
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
                        onClick={() => setInteractionMode(interactionMode === 'pan' ? null : 'pan')}
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
                      onClick={() => setInteractionMode(interactionMode === 'brightness' ? null : 'brightness')}
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
                      onClick={() => setInteractionMode(interactionMode === 'contrast' ? null : 'contrast')}
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
                padding: '20px 16px',
                background: expandedSections.tools 
                  ? 'rgba(51, 65, 85, 0.6)' 
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '15px',
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
              <div style={{ padding: '12px' }}>
                <div className="sidebar-buttons">
                  <button className={`sidebar-btn ${mode === 'select' ? 'active' : ''}`} onClick={() => setMode('select')}>Select</button>
                  <button className={`sidebar-btn ${mode === 'measure' ? 'active' : ''}`} onClick={() => setMode('measure')}>Measure</button>
                  <button className={`sidebar-btn ${mode === 'annotate' ? 'active' : ''}`} onClick={() => setMode('annotate')}>Annotate</button>
                  <button className={`sidebar-btn ${mode === 'crosshair' ? 'active' : ''}`} onClick={() => setMode('crosshair')}>Crosshair</button>
              </div>
            </div>
          )}
          </div>

          {/* TRANSFORM Section */}
          <div style={{ marginBottom: '0px' }}>
                <button
              onClick={() => toggleSection('transform')}
              style={{
                width: '100%',
                padding: '20px 16px',
                background: expandedSections.transform 
                  ? 'rgba(51, 65, 85, 0.6)' 
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '15px',
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
                {/* 3D Cube Icon */}
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                  <line x1="12" y1="22.08" x2="12" y2="12"/>
                </svg>
                <span>3D Options</span>
              </div>
              <span style={{ fontSize: '18px' }}>{expandedSections.transform ? '▼' : '▶'}</span>
            </button>
            {expandedSections.transform && (
              <div style={{ padding: '12px' }}>
                <div className="sidebar-buttons">
                  <button className="sidebar-btn" onClick={() => setFlipH(f => !f)}>Flip H</button>
                  <button className="sidebar-btn" onClick={() => setFlipV(f => !f)}>Flip V</button>
                  <button className="sidebar-btn" onClick={() => resetView()}>Reset</button>
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
                padding: '20px 16px',
                background: expandedSections.upload 
                  ? 'rgba(51, 65, 85, 0.6)' 
                  : 'rgba(51, 65, 85, 0.3)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                border: 'none',
                borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '0',
                color: '#fff',
                fontSize: '15px',
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
              <div style={{ padding: '12px' }}>
                <input id="niftiUpload" type="file" accept=".nii,.nii.gz" className="upload-box" />
              </div>
            )}
        </div>

      </div>
      </div>

    </div>
  );
};

export default DicomViewer;

