import { Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import DicomViewer from "./components/DicomViewer";
import About from "./pages/About";    

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/viewer" element={<DicomViewer />} />
      <Route path="/about" element={<About />} />
    </Routes>
  );
}
