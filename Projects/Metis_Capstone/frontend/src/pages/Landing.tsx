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

  // --- PROFESSIONAL THEME COLORS ---
  const pageBg = isDarkMode ? "#121212" : "#f7f7fb";
  const cardBg = isDarkMode ? "#1E1E1E" : "#ffffff";
  const headerBg = isDarkMode ? "#1E1E1E" : "#ffffff";
  const borderColor = isDarkMode ? "#333333" : "#e5e7eb";
  const mainText = isDarkMode ? "#E0E0E0" : "#0f172a";
  const subText = isDarkMode ? "#A0A0A0" : "#64748b";
  const accentColor = "#6366f1"; // Matches the purple/indigo from About page

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: pageBg, 
      color: mainText,
      transition: "all 0.3s ease"
    }}>
      {/* top nav */}
      <header style={{ 
        background: headerBg, 
        borderBottom: `1px solid ${borderColor}`,
        transition: "all 0.3s ease",
        position: "sticky",
        top: 0,
        zIndex: 50
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
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {/* LOGO REPLACEMENT */}
            <img 
              src="/images/Logo.jpg" 
              alt="Metis Logo" 
              style={{ 
                width: 32, 
                height: 32, 
                borderRadius: "8px",
                objectFit: "cover"
              }} 
            />
            <strong style={{ fontSize: 20 }}>Metis</strong>
          </div>

          {/* ✅ FIXED ABOUT BUTTON */}
          <nav>
            <Link 
              to="/about" 
              style={{ 
                opacity: 0.8, 
                textDecoration: "none", 
                color: "inherit",
                fontWeight: 500,
                fontSize: 16
              }}
            >
              About
            </Link>
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
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "40px 24px" }}>
        <div style={{ display: "grid", gap: 32, gridTemplateColumns: "1fr 1fr" }}>
          
          {/* left hero card */}
          <div
            style={{
              background: cardBg,
              border: `1px solid ${borderColor}`,
              borderRadius: 16,
              padding: 40,
              transition: "all 0.3s ease",
              boxShadow: isDarkMode ? "0 4px 20px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.05)"
            }}
          >
            <span
              style={{
                fontSize: 12,
                padding: "6px 12px",
                borderRadius: 9999,
                background: isDarkMode ? "rgba(99, 102, 241, 0.2)" : "#e9f5ff",
                color: isDarkMode ? "#818cf8" : "#0066cc",
                fontWeight: 600,
                border: isDarkMode ? "1px solid rgba(99, 102, 241, 0.3)" : "none"
              }}
            >
              🧠 Medical Imaging Platform
            </span>
            <h1 style={{ marginTop: 24, fontSize: 36, lineHeight: 1.2, fontWeight: 700 }}>
              Advanced DICOM Viewer
              <br />
              <span style={{ color: subText, fontWeight: 400 }}>for Medical Professionals</span>
            </h1>
            <p style={{ marginTop: 16, color: subText, fontSize: 16, lineHeight: 1.6 }}>
              Built for MRI analysis with a custom segmentation workflow. Analyze brain tumors with
              precision and speed.
            </p>
            <div style={{ marginTop: 32 }}>
              <a
                href="https://github.com/Er1ckD1az/Metis_DICOM_Viewer"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  background: isDarkMode ? "#333" : "#000",
                  color: "#fff",
                  padding: "12px 20px",
                  borderRadius: 8,
                  textDecoration: "none",
                  fontWeight: 600,
                  display: "inline-block",
                  border: isDarkMode ? "1px solid #555" : "none"
                }}
              >
                View Documentation
              </a>
            </div>
          </div>

          {/* right hero card */}
          <div
            style={{
              background: cardBg,
              border: `1px solid ${borderColor}`,
              borderRadius: 16,
              padding: 32,
              position: "relative",
              transition: "all 0.3s ease",
              boxShadow: isDarkMode ? "0 4px 20px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.05)"
            }}
          >
            <div
              style={{
                position: "absolute",
                right: 16,
                top: 16,
                fontSize: 12,
                padding: "4px 10px",
                borderRadius: 9999,
                background: isDarkMode ? "rgba(34, 197, 94, 0.2)" : "#eafbea",
                color: isDarkMode ? "#4ade80" : "#17803d",
                border: isDarkMode ? "1px solid rgba(34, 197, 94, 0.3)" : "1px solid #c7e8cf",
                zIndex: 3,
                fontWeight: 600
              }}
            >
              ✅ DICOM Compatible
            </div>

            <div
              style={{
                height: 280,
                display: "grid",
                placeItems: "center",
                backgroundImage:
                  "url('/images/radiology-workstation-about-us-1024x693.jpg')",
                backgroundSize: "cover",
                backgroundPosition: "center",
                borderRadius: 12,
                border: `1px solid ${borderColor}`,
                position: "relative",
                overflow: "hidden",
                color: "#fff",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.4)",
                  backdropFilter: "blur(2px)"
                }}
              />
              <div
                style={{
                  position: "relative",
                  textAlign: "center",
                  zIndex: 2,
                  textShadow: "0 2px 4px rgba(0,0,0,0.8)",
                }}
              >
                <div style={{ fontSize: 32 }}>🖥️</div>
                <div style={{ marginTop: 8, fontWeight: 700, fontSize: 22 }}>
                  Medical Imaging Workstation
                </div>
                <div style={{ fontSize: 14, opacity: 0.9 }}>
                  Advanced DICOM Viewer Interface
                </div>
              </div>
            </div>

            <div style={{ marginTop: 16, fontSize: 13, display: "flex", gap: 8, alignItems: "center" }}>
              <span
                style={{
                  padding: "2px 8px",
                  background: isDarkMode ? "#333" : "#f3f4f6",
                  border: `1px solid ${borderColor}`,
                  borderRadius: 6,
                  color: mainText,
                  fontWeight: 600,
                }}
              >
                3D
              </span>{" "}
              <span style={{ color: subText }}>Volume Rendering • Real-time 3D visualization</span>
            </div>
          </div>
        </div>
      </section>

      {/* upload section */}
      <section id="upload" style={{ maxWidth: 880, margin: "0 auto", padding: "0 24px 40px 24px" }}>
        <div
          style={{
            background: cardBg,
            border: `1px solid ${borderColor}`,
            borderRadius: 16,
            padding: 40,
            transition: "all 0.3s ease",
            boxShadow: isDarkMode ? "0 4px 20px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.05)"
          }}
        >
          <div style={{ textAlign: "center", maxWidth: 640, margin: "0 auto" }}>
            <h2 style={{ fontSize: 28, margin: 0, fontWeight: 700 }}>Upload Your DICOM Files</h2>
            <p style={{ marginTop: 8, color: subText }}>
              Drag & drop .dcm or .nii/.nii.gz files, or click to browse (UI demo only).
            </p>
          </div>

          <label
            style={{
              marginTop: 32,
              display: "block",
              textAlign: "center",
              border: `2px dashed ${isDarkMode ? "#444" : "#d1d5db"}`,
              borderRadius: 12,
              padding: 48,
              cursor: "pointer",
              background: isDarkMode ? "#121212" : "#fafafa",
              transition: "all 0.3s ease"
            }}
          >
            <div style={{ fontSize: 32, marginBottom: 16 }}>⬆️</div>
            <div style={{ fontWeight: 600, fontSize: 18, color: mainText }}>Upload your DICOM file</div>
            <div style={{ fontSize: 14, color: subText, marginTop: 4 }}>
              Supports NIfTI (.nii, .nii.gz) and DICOM (.dcm) up to 500MB
            </div>
            <input type="file" className="hidden" accept=".dcm,.nii,.nii.gz" onChange={handleFileUpload} />
          </label>

          <div style={{ marginTop: 24, textAlign: "center" }}>
            <label
              style={{
                display: "inline-flex",
                gap: 8,
                alignItems: "center",
                padding: "10px 20px",
                borderRadius: 8,
                border: `1px solid ${borderColor}`,
                background: isDarkMode ? "#333" : "#fff",
                color: mainText,
                cursor: "pointer",
                transition: "all 0.3s ease",
                fontWeight: 500
              }}
            >
              <input type="file" className="hidden" accept=".dcm,.nii,.nii.gz" onChange={handleFileUpload} />
              <span>Choose File</span>
            </label>
          </div>
        </div>
      </section>

      {/* experience */}
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "0 24px 40px 24px" }}>
        <div style={{ textAlign: "center", maxWidth: 760, margin: "0 auto 32px auto" }}>
          <h2 style={{ fontSize: 28, margin: 0, fontWeight: 700 }}>Experience Metis</h2>
          <p style={{ marginTop: 8, color: subText }}>
            Explore key features like 3D visualization, measurements, and annotations.
          </p>
        </div>

        <div style={{ display: "grid", gap: 32, gridTemplateColumns: "1fr 1fr" }}>
          <div
            style={{ 
              background: cardBg, 
              border: `1px solid ${borderColor}`, 
              borderRadius: 16, 
              padding: 32,
              transition: "all 0.3s ease",
              boxShadow: isDarkMode ? "0 4px 20px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.05)"
            }}
          >
            <div style={{ fontSize: 20, fontWeight: 600 }}>Try Our Demo</div>
            <p style={{ marginTop: 8, color: subText, lineHeight: 1.6 }}>
              Don't have a DICOM file? Launch the demo viewer and explore the tools.
            </p>
            <div style={{ marginTop: 24 }}>
              <Link
                to="/viewer"
                state={{ demoMode: true }}
                style={{ 
                  background: isDarkMode ? "#333" : "#000", 
                  color: "#fff", 
                  padding: "12px 20px", 
                  borderRadius: 8, 
                  textDecoration: 'none', 
                  display: 'inline-block',
                  fontWeight: 600,
                  border: isDarkMode ? "1px solid #555" : "none"
                }}
              >
                ▶︎ Launch Demo Viewer
              </Link>
            </div>
          </div>

          <div
            style={{ 
              background: cardBg, 
              border: `1px solid ${borderColor}`, 
              borderRadius: 16, 
              padding: 32,
              transition: "all 0.3s ease",
              boxShadow: isDarkMode ? "0 4px 20px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.05)"
            }}
          >
            <div
              style={{
                height: 180,
                borderRadius: 12,
                border: `1px solid ${borderColor}`,
                // UPDATED IMAGE SECTION HERE
                backgroundImage: "url('/images/Sample MRI Scan.jpg')",
                backgroundSize: "cover",
                backgroundPosition: "center",
                display: "grid",
                placeItems: "center",
                transition: "all 0.3s ease",
                position: "relative",
                overflow: "hidden",
                color: "#fff"
              }}
            >
              {/* Overlay with blur effect */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.4)",
                  backdropFilter: "blur(2px)"
                }}
              />
              
              {/* Content */}
              <div style={{ textAlign: "center", position: "relative", zIndex: 2, textShadow: "0 2px 4px rgba(0,0,0,0.8)" }}>
                <div style={{ fontSize: 32 }}>🧪</div>
                <div style={{ fontWeight: 600, marginTop: 8, color: "#fff" }}>Sample MRI Scan</div>
                <div style={{ fontSize: 14, color: "rgba(255,255,255,0.9)" }}>Brain imaging dataset</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* footer */}
      <footer style={{ 
        marginTop: 40, 
        background: headerBg, 
        borderTop: `1px solid ${borderColor}`,
        transition: "all 0.3s ease"
      }}>
        <div
          style={{
            maxWidth: 1120,
            margin: "0 auto",
            padding: "32px 24px",
            textAlign: "center",
            fontSize: 14,
            color: subText,
          }}
        >
          <div style={{ fontWeight: 700, color: mainText, marginBottom: 8, fontSize: 16 }}>Metis</div>
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