import { Link, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";

export default function Landing() {
  console.log("Landing mounted");
  const navigate = useNavigate();
  
  // Load dark mode preference from localStorage
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved === 'true';
  });

  // Save dark mode preference to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('darkMode', isDarkMode.toString());
  }, [isDarkMode]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      console.log("File selected from landing page:", file.name);
      navigate('/viewer', { state: { uploadedFile: file } });
    }
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: isDarkMode ? "#0f172a" : "#f7f7fb", 
      color: isDarkMode ? "#f1f5f9" : "#0f172a",
      transition: "all 0.3s ease"
    }}>
      {/* top nav */}
      <header style={{ 
        background: isDarkMode ? "#1e293b" : "#fff", 
        borderBottom: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
        transition: "all 0.3s ease"
      }}>
        <div
          style={{
            maxWidth: 1120,
            margin: "0 auto",
            padding: "12px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div
              style={{
                width: 24,
                height: 24,
                borderRadius: 9999,
                background: isDarkMode ? "#3b82f6" : "#000",
                color: "#fff",
                display: "grid",
                placeItems: "center",
                fontSize: 12,
                transition: "all 0.3s ease"
              }}
            >
              M
            </div>
            <strong>Metis</strong>
          </div>
          <nav>
            <a href="#about" style={{ opacity: 0.8 }}>
              About
            </a>
          </nav>
        </div>
      </header>

      {/* Dark Mode Toggle - Fixed to top right corner */}
      <div style={{
        position: "fixed",
        top: 16,
        right: 16,
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        gap: 8
      }}>
        {/* Sun icon */}
        <svg 
          width="20" 
          height="20" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke={isDarkMode ? "#64748b" : "#f59e0b"} 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round"
          style={{ transition: "all 0.3s ease" }}
        >
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

        {/* Toggle switch */}
        <button
          onClick={() => setIsDarkMode(!isDarkMode)}
          style={{
            background: isDarkMode ? "rgba(59, 130, 246, 0.2)" : "rgba(15, 23, 42, 0.1)",
            border: `1px solid ${isDarkMode ? "rgba(59, 130, 246, 0.4)" : "rgba(15, 23, 42, 0.2)"}`,
            borderRadius: 20,
            padding: "3px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            transition: "all 0.3s ease",
            position: "relative",
            width: 48,
            height: 24,
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.15)"
          }}
          title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
        >
          {/* Slider circle */}
          <div style={{
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: isDarkMode ? "#3b82f6" : "#0f172a",
            position: "absolute",
            left: isDarkMode ? 27 : 3,
            transition: "all 0.3s ease",
            boxShadow: "0 2px 4px rgba(0,0,0,0.3)"
          }} />
        </button>

        {/* Moon icon */}
        <svg 
          width="20" 
          height="20" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke={isDarkMode ? "#3b82f6" : "#64748b"} 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round"
          style={{ transition: "all 0.3s ease" }}
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      </div>

      {/* hero row */}
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "24px" }}>
        <div style={{ display: "grid", gap: 24, gridTemplateColumns: "1fr 1fr" }}>
          {/* left hero card */}
          <div
            style={{
              background: isDarkMode ? "#1e293b" : "#fff",
              border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
              borderRadius: 12,
              padding: 32,
              transition: "all 0.3s ease"
            }}
          >
            <span
              style={{
                fontSize: 12,
                padding: "4px 8px",
                borderRadius: 9999,
                background: "#e9f5ff",
                color: "#0066cc",
              }}
            >
              🧠 Medical Imaging Platform
            </span>
            <h1 style={{ marginTop: 16, fontSize: 32, lineHeight: 1.15 }}>
              Advanced DICOM Viewer
              <br />
              for Medical Professionals
            </h1>
            <p style={{ marginTop: 8, opacity: 0.8 }}>
              Built for MRI analysis with a custom segmentation workflow. Analyze brain tumors with
              precision.
            </p>
            <div style={{ marginTop: 20 }}>
              <a
                href="https://github.com/Er1ckD1az/Metis_DICOM_Viewer"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  background: "#000",
                  color: "#fff",
                  padding: "10px 14px",
                  borderRadius: 8,
                  textDecoration: "none",
                  fontWeight: 600,
                }}
              >
                View Documentation
              </a>
            </div>
          </div>

          {/* right hero card (WITH image background) */}
          <div
            style={{
              background: isDarkMode ? "#1e293b" : "#fff",
              border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
              borderRadius: 12,
              padding: 32,
              position: "relative",
              transition: "all 0.3s ease"
            }}
          >
            <div
              style={{
                position: "absolute",
                right: 16,
                top: 16,
                fontSize: 12,
                padding: "4px 8px",
                borderRadius: 9999,
                background: "#eafbea",
                color: "#17803d",
                border: "1px solid #c7e8cf",
                zIndex: 3,
              }}
            >
              ✅ DICOM Compatible
            </div>

            <div
              style={{
                height: 240,
                display: "grid",
                placeItems: "center",
                backgroundImage:
                  "url('/images/radiology-workstation-about-us-1024x693.jpg')",
                backgroundSize: "cover",
                backgroundPosition: "center",
                borderRadius: 8,
                border: "1px solid #e5e7eb",
                position: "relative",
                overflow: "hidden",
                color: "#fff",
              }}
            >
              {/* translucent overlay so text pops */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.25)",
                }}
              />
              <div
                style={{
                  position: "relative",
                  textAlign: "center",
                  zIndex: 2,
                  textShadow: "0 1px 2px rgba(0,0,0,0.6)",
                }}
              >
                <div style={{ fontSize: 28 }}>🖥️</div>
                <div style={{ marginTop: 6, fontWeight: 600, fontSize: 20 }}>
                  Medical Imaging Workstation
                </div>
                <div style={{ fontSize: 14, opacity: 0.95 }}>
                  Advanced DICOM Viewer Interface
                </div>
              </div>
            </div>

            <div style={{ marginTop: 12, fontSize: 12 }}>
              <span
                style={{
                  padding: "2px 8px",
                  background: "#f3f4f6",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                }}
              >
                3D
              </span>{" "}
              <span style={{ opacity: 0.7 }}>Volume Rendering • Real-time 3D visualization</span>
            </div>
          </div>
        </div>
      </section>

      {/* upload section */}
      <section id="upload" style={{ maxWidth: 880, margin: "0 auto", padding: "24px" }}>
        <div
          style={{
            background: isDarkMode ? "#1e293b" : "#fff",
            border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 32,
            transition: "all 0.3s ease"
          }}
        >
          <div style={{ textAlign: "center", maxWidth: 640, margin: "0 auto" }}>
            <h2 style={{ fontSize: 24, margin: 0 }}>Upload Your DICOM Files</h2>
            <p style={{ marginTop: 8, opacity: 0.8 }}>
              Drag & drop .dcm or .nii/.nii.gz files, or click to browse (UI demo only).
            </p>
          </div>

          <label
            style={{
              marginTop: 16,
              display: "block",
              textAlign: "center",
              border: isDarkMode ? "2px dashed #475569" : "2px dashed #d1d5db",
              borderRadius: 12,
              padding: 32,
              cursor: "pointer",
              background: isDarkMode ? "#0f172a" : "#fafafa",
              transition: "all 0.3s ease"
            }}
          >
            <div style={{ fontSize: 28 }}>⬆️</div>
            <div style={{ marginTop: 6, fontWeight: 600 }}>Upload your DICOM file</div>
            <div style={{ fontSize: 13, opacity: 0.7 }}>
              Supports NIfTI (.nii, .nii.gz) and DICOM (.dcm) up to 500MB
            </div>
            <input type="file" className="hidden" accept=".dcm,.nii,.nii.gz" onChange={handleFileUpload} />
          </label>

          <div style={{ marginTop: 12, textAlign: "center" }}>
            <label
              style={{
                display: "inline-flex",
                gap: 8,
                alignItems: "center",
                padding: "8px 12px",
                borderRadius: 8,
                border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
                background: isDarkMode ? "#1e293b" : "#fff",
                cursor: "pointer",
                transition: "all 0.3s ease"
              }}
            >
              <input type="file" className="hidden" accept=".dcm,.nii,.nii.gz" onChange={handleFileUpload} />
              <span>Choose File</span>
            </label>
          </div>
        </div>
      </section>

      {/* experience / demo (restored) */}
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "24px" }}>
        <div style={{ textAlign: "center", maxWidth: 760, margin: "0 auto" }}>
          <h2 style={{ fontSize: 24, margin: 0 }}>Experience Metis</h2>
          <p style={{ marginTop: 8, opacity: 0.8 }}>
            Explore key features like 3D visualization, measurements, and annotations.
          </p>
        </div>

        <div style={{ marginTop: 16, display: "grid", gap: 24, gridTemplateColumns: "1fr 1fr" }}>
          <div
            style={{ 
              background: isDarkMode ? "#1e293b" : "#fff", 
              border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb", 
              borderRadius: 12, 
              padding: 24,
              transition: "all 0.3s ease"
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 600 }}>Try Our Demo</div>
            <p style={{ marginTop: 6, opacity: 0.8 }}>
              Don't have a DICOM file? Launch the demo viewer and explore the tools.
            </p>
            <div style={{ marginTop: 12 }}>
              <Link
                to="/viewer"
                state={{ demoMode: true }}
                style={{ background: "#000", color: "#fff", padding: "10px 14px", borderRadius: 8, textDecoration: 'none', display: 'inline-block' }}
              >
                ▶︎ Launch Demo Viewer
              </Link>
            </div>
          </div>

          <div
            style={{ 
              background: isDarkMode ? "#1e293b" : "#fff", 
              border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb", 
              borderRadius: 12, 
              padding: 24,
              transition: "all 0.3s ease"
            }}
          >
            <div
              style={{
                height: 160,
                borderRadius: 8,
                border: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
                background: isDarkMode ? "linear-gradient(135deg,#1e293b,#0f172a)" : "linear-gradient(135deg,#eef2f7,#e5e7eb)",
                display: "grid",
                placeItems: "center",
                transition: "all 0.3s ease"
              }}
            >
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 22 }}>🧪</div>
                <div style={{ fontWeight: 600, marginTop: 4 }}>Sample MRI Scan</div>
                <div style={{ fontSize: 13, opacity: 0.7 }}>Brain imaging dataset</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* footer */}
      <footer style={{ 
        marginTop: 24, 
        background: isDarkMode ? "#1e293b" : "#fff", 
        borderTop: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
        transition: "all 0.3s ease"
      }}>
        <div
          style={{
            maxWidth: 1120,
            margin: "0 auto",
            padding: "24px",
            textAlign: "center",
            fontSize: 14,
            opacity: 0.85,
          }}
        >
          <div style={{ fontWeight: 600 }}>Metis</div>
          <div>
            Professional DICOM viewing platform for medical imaging professionals worldwide.
          </div>
        </div>
      </footer>

      {/* build badge */}
      <div
        style={{
          position: "fixed",
          right: 12,
          bottom: 12,
          background: "#000",
          color: "#fff",
          padding: "6px 10px",
          borderRadius: 8,
          fontSize: 12,
          opacity: 0.85,
          zIndex: 9999,
        }}
      >
        build: landing v3
      </div>
    </div>
  );
}