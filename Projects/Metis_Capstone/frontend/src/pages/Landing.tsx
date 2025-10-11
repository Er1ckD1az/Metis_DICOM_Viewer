import { Link, useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Landing() {
  console.log("Landing mounted");
  const [darkMode, setDarkMode] = useState(false);
  const navigate = useNavigate();

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    if (file.name.toLowerCase().endsWith('.nii') || 
        file.name.toLowerCase().endsWith('.nii.gz') ||
        file.name.toLowerCase().endsWith('.dcm')) {
      // Navigate to viewer with the file
      navigate('/viewer', { state: { uploadedFile: file } });
    } else {
      alert('Please upload a valid DICOM (.dcm) or NIfTI (.nii, .nii.gz) file');
    }
  };

  const colors = darkMode
    ? {
        bg: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
        cardBg: "rgba(30, 41, 59, 0.4)",
        text: "#f1f5f9",
        border: "rgba(148, 163, 184, 0.2)",
        headerBg: "rgba(30, 41, 59, 0.6)",
        mutedText: "rgba(241, 245, 249, 0.8)",
        tagBg: "rgba(30, 58, 95, 0.5)",
        tagColor: "#93c5fd",
        shadow: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
      }
    : {
        bg: "linear-gradient(135deg, #e0e7ff 0%, #f7f7fb 50%, #fce7f3 100%)",
        cardBg: "rgba(255, 255, 255, 0.4)",
        text: "#0f172a",
        border: "rgba(226, 232, 240, 0.3)",
        headerBg: "rgba(255, 255, 255, 0.6)",
        mutedText: "rgba(15, 23, 42, 0.8)",
        tagBg: "rgba(233, 245, 255, 0.6)",
        tagColor: "#0066cc",
        shadow: "0 8px 32px 0 rgba(31, 38, 135, 0.15)",
      };

  return (
    <div style={{ minHeight: "100vh", background: colors.bg, color: colors.text, transition: "all 0.3s ease" }}>
      {/* top nav */}
      <header style={{ 
        background: colors.headerBg, 
        borderBottom: `1px solid ${colors.border}`, 
        transition: "all 0.3s ease", 
        position: "relative",
        backdropFilter: "blur(10px)",
        WebkitBackdropFilter: "blur(10px)",
        boxShadow: colors.shadow,
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
                background: darkMode ? "#fff" : "#000",
                color: darkMode ? "#000" : "#fff",
                display: "grid",
                placeItems: "center",
                fontSize: 12,
                transition: "all 0.3s ease",
              }}
            >
              M
            </div>
            <strong>Metis</strong>
          </div>
          <nav style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <a href="#about" style={{ opacity: 0.8 }}>
              About
            </a>
          </nav>
        </div>
        {/* Dark Mode Toggle - positioned at far right */}
        <div
          style={{
            position: "absolute",
            right: 24,
            top: "50%",
            transform: "translateY(-50%)",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          {/* Sun Icon */}
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke={darkMode ? colors.mutedText : colors.text}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ transition: "all 0.3s ease", opacity: darkMode ? 0.5 : 1 }}
          >
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </svg>

          <button
            onClick={() => setDarkMode(!darkMode)}
            style={{
              width: 48,
              height: 24,
              borderRadius: 12,
              background: darkMode ? "#3b82f6" : "#cbd5e1",
              border: "none",
              cursor: "pointer",
              position: "relative",
              transition: "background 0.3s ease",
              padding: 0,
            }}
            aria-label="Toggle dark mode"
          >
            <div
              style={{
                width: 20,
                height: 20,
                borderRadius: "50%",
                background: "#fff",
                position: "absolute",
                top: 2,
                left: darkMode ? 26 : 2,
                transition: "left 0.3s ease",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
              }}
            />
          </button>

          {/* Moon Icon */}
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke={darkMode ? colors.text : colors.mutedText}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ transition: "all 0.3s ease", opacity: darkMode ? 1 : 0.5 }}
          >
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        </div>
      </header>

      {/* hero row */}
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "24px" }}>
        <div style={{ display: "grid", gap: 24, gridTemplateColumns: "1fr 1fr" }}>
          {/* left hero card */}
          <div
            style={{
              background: colors.cardBg,
              border: `1px solid ${colors.border}`,
              borderRadius: 12,
              padding: 32,
              transition: "all 0.3s ease",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              boxShadow: colors.shadow,
            }}
          >
            <span
              style={{
                fontSize: 12,
                padding: "4px 8px",
                borderRadius: 9999,
                background: colors.tagBg,
                color: colors.tagColor,
                transition: "all 0.3s ease",
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
                border: `1px solid ${colors.border}`,
              }}
            >
              🧠 Medical Imaging Platform
            </span>
            <h1 style={{ marginTop: 16, fontSize: 32, lineHeight: 1.15 }}>
              Advanced DICOM Viewer
              <br />
              for Medical Professionals
            </h1>
            <p style={{ marginTop: 8, color: colors.mutedText }}>
              Built for MRI analysis with a custom segmentation workflow. Analyze brain tumors with
              precision.
            </p>
            <div style={{ marginTop: 20 }}>
              <a
                href="https://github.com/Er1ckD1az/Metis_DICOM_Viewer"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  background: darkMode ? "#3b82f6" : "#000",
                  color: "#fff",
                  padding: "10px 14px",
                  borderRadius: 8,
                  textDecoration: "none",
                  fontWeight: 600,
                  display: "inline-block",
                  transition: "all 0.3s ease",
                }}
              >
                View Documentation
              </a>
            </div>
          </div>

          {/* right hero card (WITH image background) */}
          <div
            style={{
              background: colors.cardBg,
              border: `1px solid ${colors.border}`,
              borderRadius: 12,
              padding: 32,
              position: "relative",
              transition: "all 0.3s ease",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              boxShadow: colors.shadow,
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
                background: darkMode ? "rgba(30, 77, 43, 0.6)" : "rgba(234, 251, 234, 0.8)",
                color: darkMode ? "#86efac" : "#17803d",
                border: `1px solid ${darkMode ? "#15803d" : "#c7e8cf"}`,
                zIndex: 3,
                transition: "all 0.3s ease",
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
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
                border: `1px solid ${colors.border}`,
                position: "relative",
                overflow: "hidden",
                color: "#fff",
                transition: "all 0.3s ease",
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
                  background: darkMode ? "rgba(51, 65, 85, 0.6)" : "rgba(243, 244, 246, 0.8)",
                  border: `1px solid ${colors.border}`,
                  borderRadius: 6,
                  transition: "all 0.3s ease",
                  backdropFilter: "blur(10px)",
                  WebkitBackdropFilter: "blur(10px)",
                }}
              >
                3D
              </span>{" "}
              <span style={{ color: colors.mutedText }}>Volume Rendering • Real-time 3D visualization</span>
            </div>
          </div>
        </div>
      </section>

      {/* upload section */}
      <section id="upload" style={{ maxWidth: 880, margin: "0 auto", padding: "24px" }}>
        <div
          style={{
            background: colors.cardBg,
            border: `1px solid ${colors.border}`,
            borderRadius: 12,
            padding: 32,
            transition: "all 0.3s ease",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            boxShadow: colors.shadow,
          }}
        >
          <div style={{ textAlign: "center", maxWidth: 640, margin: "0 auto" }}>
            <h2 style={{ fontSize: 24, margin: 0 }}>Upload Your DICOM Files</h2>
            <p style={{ marginTop: 8, color: colors.mutedText }}>
              Drag & drop .dcm or .nii/.nii.gz files, or click to browse (UI demo only).
            </p>
          </div>

          <label
            style={{
              marginTop: 16,
              display: "block",
              textAlign: "center",
              border: `2px dashed ${colors.border}`,
              borderRadius: 12,
              padding: 32,
              cursor: "pointer",
              background: darkMode ? "rgba(15, 23, 42, 0.3)" : "rgba(250, 250, 250, 0.5)",
              transition: "all 0.3s ease",
              backdropFilter: "blur(10px)",
              WebkitBackdropFilter: "blur(10px)",
            }}
          >
            <div style={{ fontSize: 28 }}>⬆️</div>
            <div style={{ marginTop: 6, fontWeight: 600 }}>Upload your DICOM file</div>
            <div style={{ fontSize: 13, color: colors.mutedText }}>
              Supports NIfTI (.nii, .nii.gz) and DICOM (.dcm) up to 500MB
            </div>
            <input 
              type="file" 
              className="hidden" 
              accept=".dcm,.nii,.nii.gz" 
              onChange={handleFileUpload}
            />
          </label>

          <div style={{ marginTop: 12, textAlign: "center" }}>
            <label
              style={{
                display: "inline-flex",
                gap: 8,
                alignItems: "center",
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${colors.border}`,
                background: colors.cardBg,
                cursor: "pointer",
                transition: "all 0.3s ease",
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
              }}
            >
              <input 
                type="file" 
                className="hidden" 
                accept=".dcm,.nii,.nii.gz"
                onChange={handleFileUpload}
              />
              <span>Choose File</span>
            </label>
          </div>
        </div>
      </section>

      {/* experience / demo (restored) */}
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "24px" }}>
        <div style={{ textAlign: "center", maxWidth: 760, margin: "0 auto" }}>
          <h2 style={{ fontSize: 24, margin: 0 }}>Experience Metis</h2>
          <p style={{ marginTop: 8, color: colors.mutedText }}>
            Explore key features like 3D visualization, measurements, and annotations.
          </p>
        </div>

        <div style={{ marginTop: 16, display: "grid", gap: 24, gridTemplateColumns: "1fr 1fr" }}>
          <div
            style={{ 
              background: colors.cardBg, 
              border: `1px solid ${colors.border}`, 
              borderRadius: 12, 
              padding: 24,
              transition: "all 0.3s ease",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              boxShadow: colors.shadow,
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 600 }}>Try Our Demo</div>
            <p style={{ marginTop: 6, color: colors.mutedText }}>
              Don't have a DICOM file? Launch the demo viewer and explore the tools.
            </p>
            <div style={{ marginTop: 12 }}>
              <Link
                to="/viewer"
                state={{ demoMode: true }}
                style={{ 
                  background: darkMode ? "#3b82f6" : "#000", 
                  color: "#fff", 
                  padding: "10px 14px", 
                  borderRadius: 8,
                  display: "inline-block",
                  transition: "all 0.3s ease",
                }}
              >
                ▶︎ Launch Demo Viewer
              </Link>
            </div>
          </div>

          <div
            style={{ 
              background: colors.cardBg, 
              border: `1px solid ${colors.border}`, 
              borderRadius: 12, 
              padding: 24,
              transition: "all 0.3s ease",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              boxShadow: colors.shadow,
            }}
          >
            <div
              style={{
                height: 160,
                borderRadius: 8,
                border: `1px solid ${colors.border}`,
                background: darkMode 
                  ? "rgba(30, 41, 59, 0.5)" 
                  : "rgba(238, 242, 247, 0.6)",
                display: "grid",
                placeItems: "center",
                transition: "all 0.3s ease",
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
              }}
            >
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 22 }}>🧪</div>
                <div style={{ fontWeight: 600, marginTop: 4 }}>Sample MRI Scan</div>
                <div style={{ fontSize: 13, color: colors.mutedText }}>Brain imaging dataset</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* footer */}
      <footer style={{ 
        marginTop: 24, 
        background: colors.headerBg, 
        borderTop: `1px solid ${colors.border}`,
        transition: "all 0.3s ease",
        backdropFilter: "blur(10px)",
        WebkitBackdropFilter: "blur(10px)",
        boxShadow: colors.shadow,
      }}>
        <div
          style={{
            maxWidth: 1120,
            margin: "0 auto",
            padding: "24px",
            textAlign: "center",
            fontSize: 14,
            color: colors.mutedText,
          }}
        >
          <div style={{ fontWeight: 600, color: colors.text }}>Metis</div>
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
          background: darkMode ? "rgba(30, 41, 59, 0.6)" : "rgba(0, 0, 0, 0.6)",
          color: "#fff",
          padding: "6px 10px",
          borderRadius: 8,
          fontSize: 12,
          zIndex: 9999,
          transition: "all 0.3s ease",
          backdropFilter: "blur(10px)",
          WebkitBackdropFilter: "blur(10px)",
          border: `1px solid ${colors.border}`,
        }}
      >
        build: landing v3
      </div>
    </div>
  );
}
