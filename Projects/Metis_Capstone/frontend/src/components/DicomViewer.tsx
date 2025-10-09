import React, { useEffect, useRef, useState } from "react";
import "./DicomViewer.css";
import * as cornerstone from "cornerstone-core";
import * as cornerstoneTools from "cornerstone-tools";
import * as cornerstoneWADOImageLoader from "cornerstone-wado-image-loader";
import Hammer from "hammerjs";

// wire up external deps for cornerstone-tools & wado loader
cornerstoneTools.external.cornerstone = cornerstone as any;
cornerstoneTools.external.Hammer = Hammer as any;
cornerstoneWADOImageLoader.external.cornerstone = cornerstone as any;

const DicomViewer: React.FC = () => {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const [hasImage, setHasImage] = useState(false);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    // enable the viewport
    cornerstone.enable(element);

    const handleFileChange = async (e: Event) => {
      const input = e.target as HTMLInputElement;
      const file = input?.files?.[0];
      if (!file) return;

      try {
        // Cornerstone WADO loader is DICOM-only (.dcm). NIfTI will not render here.
        const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
        const image = await cornerstone.loadImage(imageId);
        cornerstone.displayImage(element, image);
        setHasImage(true);
      } catch (error) {
        console.error("Failed to load DICOM image:", error);
        setHasImage(false);
      }
    };

    const fileInput = document.getElementById("dicomUpload");
    fileInput?.addEventListener("change", handleFileChange);

    // tools
    cornerstoneTools.init();
    const WwwcTool = (cornerstoneTools as any).WwwcTool;
    const PanTool = (cornerstoneTools as any).PanTool;
    const ZoomTool = (cornerstoneTools as any).ZoomTool;

    cornerstoneTools.addTool(WwwcTool);
    cornerstoneTools.addTool(PanTool);
    cornerstoneTools.addTool(ZoomTool);
    cornerstoneTools.setToolActive("Wwwc", { mouseButtonMask: 1 });

    return () => {
      fileInput?.removeEventListener("change", handleFileChange);
      cornerstone.disable(element);
    };
  }, []);

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
                <p>Upload a DICOM file to begin viewing</p>
              </div>
            </div>
          )}
          <span className="orientation top">A</span>
          <span className="orientation left">R</span>
          <span className="orientation right">L</span>
        </div>
      </div>

      {/* Right sidebar */}
      <div className="viewer-sidebar">
        <div>
          <div className="sidebar-title">DICOM Viewer</div>

          <div className="sidebar-section">
            <div className="sidebar-buttons">
              <button className="sidebar-btn">Home</button>
              <button className="sidebar-btn">Data</button>
              <button className="sidebar-btn">Search</button>
              <button className="sidebar-btn">Brightness</button>
            </div>
          </div>

          <p style={{ fontSize: "11px", color: "#aaa", marginBottom: "5px" }}>
            VIEWING
          </p>
          <div className="sidebar-buttons">
            <button className="sidebar-btn active">Select</button>
            <button className="sidebar-btn">Zoom</button>
            <button className="sidebar-btn">Pan</button>
            <button className="sidebar-btn active">Windowing</button>
            <button className="sidebar-btn">Reset</button>
            <button className="sidebar-btn">Fullscreen</button>
          </div>

          <div className="sidebar-section" style={{ marginTop: "15px" }}>
            <div className="sidebar-buttons">
              <button className="sidebar-btn">Flip H</button>
              <button className="sidebar-btn">Flip V</button>
              <button className="sidebar-btn">Settings</button>
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-buttons">
              <button className="sidebar-btn">Measure</button>
              <button className="sidebar-btn">Annotate</button>
              <button className="sidebar-btn">Crosshair</button>
              <button className="sidebar-btn">Screenshot</button>
            </div>
          </div>
        </div>

        <input
          id="dicomUpload"
          type="file"
          accept=".dcm,.nii,.nii.gz"
          className="upload-box"
        />
      </div>
    </div>
  );
};

export default DicomViewer;
