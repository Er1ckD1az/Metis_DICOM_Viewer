import React, { useEffect, useRef } from "react";
import "./DicomViewer.css";
import * as cornerstone from "cornerstone-core";
import * as cornerstoneTools from "cornerstone-tools";
import * as cornerstoneWADOImageLoader from "cornerstone-wado-image-loader";
import Hammer from "hammerjs";

cornerstoneTools.external.cornerstone = cornerstone;
cornerstoneTools.external.Hammer = Hammer;
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;

const DicomViewer: React.FC = () => {
  const elementRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    cornerstone.enable(element);

    const handleFileChange = async (e: Event) => {
      const input = e.target as HTMLInputElement;
      const file = input?.files?.[0];
      if (!file) return;
      const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
      const image = await cornerstone.loadImage(imageId);
      cornerstone.displayImage(element, image);
    };

    const fileInput = document.getElementById("dicomUpload");
    fileInput?.addEventListener("change", handleFileChange);

    cornerstoneTools.init();
    const WwwcTool = cornerstoneTools.WwwcTool;
    const PanTool = cornerstoneTools.PanTool;
    const ZoomTool = cornerstoneTools.ZoomTool;

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
      {/* Left side viewer */}
      <div className="viewer-main">
        <div className="viewer-header">
          <div>Series: Axial Series 1 of 1</div>
          <div>Image: 108 of 150 (69.2%)</div>
        </div>

        <div className="viewer-screen" ref={elementRef}>
          <p>
            No Image Loaded
            <br />
            <small>Upload a DICOM file to begin viewing</small>
          </p>
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
          accept=".dcm"
          className="upload-box"
        />
      </div>
    </div>
  );
};

export default DicomViewer;
