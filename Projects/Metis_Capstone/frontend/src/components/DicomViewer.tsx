import React, { useEffect, useRef, useState } from "react";
import "./DicomViewer.css";
import * as nifti from "nifti-reader-js";

const DicomViewer: React.FC = () => {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hasImage, setHasImage] = useState(false);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [maxSlices, setMaxSlices] = useState(0);
  const [currentView, setCurrentView] = useState<'axial' | 'coronal' | 'sagittal'>('axial');
  const [niftiData, setNiftiData] = useState<Float32Array | null>(null);
  const [dimensions, setDimensions] = useState<[number, number, number]>([0, 0, 0]);
  const [windowLevel, setWindowLevel] = useState(200);
  const [windowWidth, setWindowWidth] = useState(600);

  // Load and display NIfTI file with improved quality
  const loadNiftiFile = async (file: File) => {
    try {
      console.log("Loading NIfTI file with improved quality:", file.name);

      const arrayBuffer = await file.arrayBuffer();

      // Check if it's a valid NIfTI file
      if (!nifti.isNIFTI(arrayBuffer)) {
        throw new Error("Not a valid NIfTI file");
      }

      // Read NIfTI header
      const header = nifti.readHeader(arrayBuffer);
      const dims = [header.dims[1], header.dims[2], header.dims[3]]; // X, Y, Z

      // Read the data
      const dataBuffer = nifti.readImage(header, arrayBuffer);

      // Convert ArrayBuffer to appropriate typed array based on data type
      let data;
      const datatype = (header as any).datatype || 4; // Default to Int16
      if (datatype === 4) { // 16-bit signed integer
        const int16Data = new Int16Array(dataBuffer);
        data = new Float32Array(int16Data);
      } else if (datatype === 8) { // 32-bit signed integer
        const int32Data = new Int32Array(dataBuffer);
        data = new Float32Array(int32Data);
      } else if (datatype === 16) { // 32-bit float
        data = new Float32Array(dataBuffer);
      } else {
        // Default to Int16 for most medical images
        const int16Data = new Int16Array(dataBuffer);
        data = new Float32Array(int16Data);
      }

      console.log("NIfTI loaded successfully:", { dims, dataLength: data.length });

      // Apply rotation and flip to the entire 3D volume for better orientation
      const rotatedData = new Float32Array(data.length);
      const [x, y, z] = dims;

      // Rotate and flip the entire volume: (i,j,k) -> (j, x-1-i, k) then flip vertically
      for (let k = 0; k < z; k++) {
        for (let j = 0; j < y; j++) {
          for (let i = 0; i < x; i++) {
            const originalIndex = i + j * x + k * x * y;
            const newI = j;
            const newJ = x - 1 - i;
            const newK = k;
            // Flip vertically: top becomes bottom
            const flippedJ = y - 1 - newJ;
            const newIndex = newI + flippedJ * y + newK * y * x;
            rotatedData[newIndex] = data[originalIndex];
          }
        }
      }

      setNiftiData(rotatedData);
      setDimensions([y, x, z] as [number, number, number]); // Swap x and y dimensions
      setMaxSlices(z - 1);
      setCurrentSlice(Math.floor(z / 2));
      setHasImage(true);

      console.log("NIfTI file loaded successfully with improved quality");
    } catch (error) {
      console.error("Failed to load NIfTI file:", error);
      alert("Failed to load NIfTI file. Please check the file format.");
    }
  };

  // Extract slice from 3D volume with improved quality
  const getSlice = (data: Float32Array, dims: [number, number, number], sliceIndex: number, view: string) => {
    const [x, y, z] = dims;
    console.log("Extracting slice with improved quality:", { x, y, z, sliceIndex, view, dataLength: data.length });

    if (view === 'axial') {
      // Axial: slice through Z dimension (horizontal slices)
      const slice = new Float32Array(x * y);
      for (let j = 0; j < y; j++) {
        for (let i = 0; i < x; i++) {
          const dataIndex = i + j * x + sliceIndex * x * y;
          if (dataIndex < data.length) {
            slice[i + j * x] = data[dataIndex];
          }
        }
      }
      return { slice, width: x, height: y };
    } else if (view === 'coronal') {
      // Coronal: slice through Y dimension (front-to-back) with right-left flip
      const slice = new Float32Array(x * z);
      for (let k = 0; k < z; k++) {
        for (let i = 0; i < x; i++) {
          const dataIndex = i + sliceIndex * x + k * x * y;
          if (dataIndex < data.length) {
            // Flip right to left: (i,k) -> (i, z-1-k)
            const flippedK = z - 1 - k;
            slice[i + flippedK * x] = data[dataIndex];
          }
        }
      }
      return { slice, width: x, height: z };
    } else {
      // Sagittal: slice through X dimension (left-to-right) with vertical flip
      const slice = new Float32Array(y * z);
      for (let k = 0; k < z; k++) {
        for (let j = 0; j < y; j++) {
          const dataIndex = sliceIndex + j * x + k * x * y;
          if (dataIndex < data.length) {
            // Flip vertically: top becomes bottom (j,k) -> (j, z-1-k)
            const flippedK = z - 1 - k;
            slice[j + flippedK * y] = data[dataIndex];
          }
        }
      }
      return { slice, width: y, height: z };
    }
  };

  // Render slice to canvas with improved quality
  const renderSlice = () => {
    if (!niftiData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { slice, width, height } = getSlice(niftiData, dimensions, currentSlice, currentView);

    console.log("Rendering slice with improved quality:", { width, height, sliceLength: slice.length, currentSlice, currentView });

    // Set canvas size to fill the entire viewer screen with high quality
    const viewerWidth = 800; // Match the viewer window size
    const viewerHeight = 600;
    const scale = Math.max(viewerWidth / width, viewerHeight / height); // Use Math.max to fill completely
    const displayWidth = Math.floor(width * scale);
    const displayHeight = Math.floor(height * scale);

    // High DPI rendering for better quality
    const dpr = window.devicePixelRatio || 1;
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.objectFit = 'cover'; // Use 'cover' to fill completely
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.right = '0';
    canvas.style.bottom = '0';
    canvas.style.imageRendering = 'high-quality';

    // Scale the context for high DPI
    if (ctx) {
      ctx.scale(dpr, dpr);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
    }

    // Apply window/level for medical imaging
    const wlMin = windowLevel - windowWidth / 2;
    const wlMax = windowLevel + windowWidth / 2;

    // Apply window/level with subtle contrast enhancement
    const windowed = slice.map(val => {
      if (val < wlMin) return 0; // Black background
      if (val > wlMax) return 255; // White for high values
      const normalized = (val - wlMin) / (wlMax - wlMin);
      // Subtle contrast enhancement - not too aggressive
      const enhanced = Math.pow(normalized, 0.9) * 255;
      return Math.floor(enhanced);
    });

    console.log("Windowed range:", {
      wlMin, wlMax,
      windowedMin: Math.min(...windowed),
      windowedMax: Math.max(...windowed)
    });

    // Create image data with high-quality interpolation
    const canvasImageData = ctx.createImageData(displayWidth, displayHeight);

    for (let y = 0; y < displayHeight; y++) {
      for (let x = 0; x < displayWidth; x++) {
        // High quality interpolation for better image quality
        const srcX = x / scale;
        const srcY = y / scale;

        // Bilinear interpolation for smoother scaling
        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        const x2 = Math.min(x1 + 1, width - 1);
        const y2 = Math.min(y1 + 1, height - 1);

        const fx = srcX - x1;
        const fy = srcY - y1;

        const val1 = windowed[x1 + y1 * width] || 0;
        const val2 = windowed[x2 + y1 * width] || 0;
        const val3 = windowed[x1 + y2 * width] || 0;
        const val4 = windowed[x2 + y2 * width] || 0;

        // High-quality bilinear interpolation for better image quality
        const val = Math.floor(
          val1 * (1 - fx) * (1 - fy) +
          val2 * fx * (1 - fy) +
          val3 * (1 - fx) * fy +
          val4 * fx * fy
        );

        const pixelIndex = (y * displayWidth + x) * 4;
        canvasImageData.data[pixelIndex] = val;     // R
        canvasImageData.data[pixelIndex + 1] = val; // G
        canvasImageData.data[pixelIndex + 2] = val; // B
        canvasImageData.data[pixelIndex + 3] = 255; // A
      }
    }

    ctx.putImageData(canvasImageData, 0, 0);
    console.log("Canvas rendered successfully with improved quality");
  };

  // Handle file upload
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Debugging: make sure FastAPI will receive a proper File
    console.log("Selected file:", file);
    console.log("Is real File object?", file instanceof File); // must be true

    if (file.name.toLowerCase().endsWith('.nii') || file.name.toLowerCase().endsWith('.nii.gz')) {
      await loadNiftiFile(file);
    } else {
      console.error("Please upload a .nii or .nii.gz file");
    }
  };

  // Update slice when currentSlice or currentView changes
  useEffect(() => {
    if (hasImage && niftiData) {
      renderSlice();
    }
  }, [currentSlice, currentView, niftiData, windowLevel, windowWidth]);

  // Handle slice navigation
  const changeSlice = (direction: 'prev' | 'next') => {
    if (direction === 'prev' && currentSlice > 0) {
      setCurrentSlice(currentSlice - 1);
    } else if (direction === 'next' && currentSlice < maxSlices) {
      setCurrentSlice(currentSlice + 1);
    }
  };

  // Handle view switching
  const switchView = (view: 'axial' | 'coronal' | 'sagittal') => {
    setCurrentView(view);

    // Update max slices based on the view and start at middle slice
    if (view === 'axial') {
      const maxSlices = dimensions[2] - 1;
      setMaxSlices(maxSlices);
      setCurrentSlice(Math.floor(maxSlices / 2));
    } else if (view === 'coronal') {
      const maxSlices = dimensions[1] - 1;
      setMaxSlices(maxSlices);
      setCurrentSlice(Math.floor(maxSlices / 2));
    } else {
      const maxSlices = dimensions[0] - 1;
      setMaxSlices(maxSlices);
      setCurrentSlice(Math.floor(maxSlices / 2));
    }

    console.log("Switched to view:", view, "Max slices:", maxSlices);
  };

  return (
    <div className="viewer-container">
      {/* Quick nav back to landing */}
      <div style={{ position: "fixed", top: 10, left: 10, zIndex: 50 }}>
        <a href="#/" style={{ color: "#9cc3ff", textDecoration: "none" }}>
          ← Back to Home
        </a>
      </div>

      {/* Left side viewer */}
      <div className="viewer-main">
        <div className="viewer-header">
          <div>Series: Axial Series 1 of 1</div>
          <div>Image: 108 of 150 (69.2%)</div>
        </div>

        <div className="viewer-screen" ref={elementRef}>
          {!hasImage && (
            <div className="empty-overlay">
              <div className="empty-box">
                <h3>No Image Loaded</h3>
                <p>Upload a .nii file to begin viewing</p>
              </div>
            </div>
          )}
          {hasImage && (
            <canvas
              ref={canvasRef}
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain'
              }}
            />
          )}
          <span className="orientation top">A</span>
          <span className="orientation left">R</span>
          <span className="orientation right">L</span>
        </div>
      </div>

      {/* Right sidebar */}
      <div className="viewer-sidebar">
        <div>
          <div className="sidebar-title">NIfTI Viewer</div>

          {/* View Selection */}
          <div className="sidebar-section">
            <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
              VIEW
            </p>
            <div className="sidebar-buttons">
              <button
                className={`sidebar-btn ${currentView === 'axial' ? 'active' : ''}`}
                onClick={() => switchView('axial')}
              >
                Axial
              </button>
              <button
                className={`sidebar-btn ${currentView === 'coronal' ? 'active' : ''}`}
                onClick={() => switchView('coronal')}
              >
                Coronal
              </button>
              <button
                className={`sidebar-btn ${currentView === 'sagittal' ? 'active' : ''}`}
                onClick={() => switchView('sagittal')}
              >
                Sagittal
              </button>
            </div>
          </div>

          {/* Slice Navigation */}
          {hasImage && (
            <div className="sidebar-section">
          <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
                SLICE NAVIGATION
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                <button
                  className="sidebar-btn"
                  onClick={() => changeSlice('prev')}
                  disabled={currentSlice === 0}
                >
                  ←
                </button>
                <span style={{ fontSize: '12px', color: '#ccc' }}>
                  {currentSlice + 1} / {maxSlices + 1}
                </span>
                <button
                  className="sidebar-btn"
                  onClick={() => changeSlice('next')}
                  disabled={currentSlice === maxSlices}
                >
                  →
                </button>
              </div>
              <input
                type="range"
                min="0"
                max={maxSlices}
                value={currentSlice}
                onChange={(e) => setCurrentSlice(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
          </div>
          )}

          {/* Window/Level Controls */}
          {hasImage && (
            <div className="sidebar-section">
              <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
                WINDOW/LEVEL
              </p>
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '10px', color: '#ccc' }}>Level:</label>
                <input
                  type="range"
                  min="-1000"
                  max="1000"
                  value={windowLevel}
                  onChange={(e) => setWindowLevel(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
                <span style={{ fontSize: '10px', color: '#ccc' }}>{windowLevel}</span>
              </div>
              <div>
                <label style={{ fontSize: '10px', color: '#ccc' }}>Width:</label>
                <input
                  type="range"
                  min="100"
                  max="2000"
                  value={windowWidth}
                  onChange={(e) => setWindowWidth(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
                <span style={{ fontSize: '10px', color: '#ccc' }}>{windowWidth}</span>
              </div>
            </div>
          )}

          {/* File Upload */}
          <div className="sidebar-section">
            <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
              UPLOAD
            </p>
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={handleFileChange}
              className="upload-box"
            />
          </div>
          {/* Detection Button */}
            {hasImage && (
              <div className="sidebar-section">
                <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
                  DETECTION
                </p>
                <button
                  className="sidebar-btn"
                  onClick={async () => {
                    const fileInput = document.querySelector(
                      'input[type="file"]'
                    ) as HTMLInputElement;
                    const file = fileInput?.files?.[0];
                    if (!file) {
                      alert("Please upload a file first.");
                      return;
                    }

                    const formData = new FormData();
                    formData.append("file", file);

                    try {
                      const response = await fetch("http://localhost:8000/mri/detect", {
                        method: "POST",
                        body: formData,
                        mode: "cors",
                      });
                      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

                      const result = await response.json();
                      console.log("Detection result:", result);

                      // Show the first detected slice
                      const firstSlice = result.slices[0];
                      const img = new Image();
                      img.src = "data:image/png;base64," + firstSlice;
                      const canvas = canvasRef.current;
                      if (canvas) {
                        const ctx = canvas.getContext("2d");
                        if (ctx) {
                          ctx.clearRect(0, 0, canvas.width, canvas.height);
                          img.onload = () =>
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        }
                      }
                    } catch (err) {
                      console.error(err);
                      alert("Error running detection.");
                    }
                  }}
                >
                  Run Detection
                </button>
              </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default DicomViewer;