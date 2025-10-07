import React, { useEffect, useRef } from "react";
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

    // Initialize tools
    cornerstoneTools.init();
    const WwwcTool = cornerstoneTools.WwwcTool;
    const PanTool = cornerstoneTools.PanTool;
    const ZoomTool = cornerstoneTools.ZoomTool;

    cornerstoneTools.addTool(WwwcTool);
    cornerstoneTools.addTool(PanTool);
    cornerstoneTools.addTool(ZoomTool);

    cornerstoneTools.setToolActive("Wwwc", { mouseButtonMask: 1 });
    cornerstoneTools.setToolActive("Pan", { mouseButtonMask: 2 });
    cornerstoneTools.setToolActive("Zoom", { mouseButtonMask: 4 });

    return () => {
      fileInput?.removeEventListener("change", handleFileChange);
      cornerstone.disable(element);
    };
  }, []);

  return (
    <div className="flex flex-col items-center min-h-screen">
      <header className="w-full text-center py-4 text-xl font-semibold bg-[#121a2b] shadow-lg">
        Brain MRI DICOM Viewer
      </header>

      <div className="mt-6">
        <input
          id="dicomUpload"
          type="file"
          accept=".dcm"
          className="mb-4 bg-gray-700 px-4 py-2 rounded-lg cursor-pointer"
        />

        <div
          ref={elementRef}
          style={{
            width: "512px",
            height: "512px",
            border: "2px dashed #555",
            backgroundColor: "black",
            position: "relative",
          }}
        >
          <p
            style={{
              color: "#888",
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
            }}
          >
            No Image Loaded
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3">
        <button className="px-4 py-2 bg-blue-600 rounded-lg">Window</button>
        <button className="px-4 py-2 bg-gray-700 rounded-lg">Pan</button>
        <button className="px-4 py-2 bg-gray-700 rounded-lg">Zoom</button>
        <button className="px-4 py-2 bg-red-600 rounded-lg">Reset</button>
      </div>
    </div>
  );
};

export default DicomViewer;
