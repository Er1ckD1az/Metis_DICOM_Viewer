import { Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import DicomViewer from "./components/DicomViewer";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/viewer" element={<DicomViewer />} />
    </Routes>
  );
}
