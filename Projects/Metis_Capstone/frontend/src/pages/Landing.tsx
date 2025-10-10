import { Link } from "react-router-dom";

export default function Landing() {
  console.log("Landing mounted");

  return (
    <div style={{ minHeight: "100vh", background: "#f7f7fb", color: "#0f172a" }}>
      {/* top nav */}
      <header style={{ background: "#fff", borderBottom: "1px solid #e5e7eb" }}>
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
                background: "#000",
                color: "#fff",
                display: "grid",
                placeItems: "center",
                fontSize: 12,
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

      {/* hero row */}
      <section style={{ maxWidth: 1120, margin: "0 auto", padding: "24px" }}>
        <div style={{ display: "grid", gap: 24, gridTemplateColumns: "1fr 1fr" }}>
          {/* left hero card */}
          <div
            style={{
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              padding: 32,
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
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              padding: 32,
              position: "relative",
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
            background: "#fff",
            border: "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 32,
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
              border: "2px dashed #d1d5db",
              borderRadius: 12,
              padding: 32,
              cursor: "pointer",
              background: "#fafafa",
            }}
          >
            <div style={{ fontSize: 28 }}>⬆️</div>
            <div style={{ marginTop: 6, fontWeight: 600 }}>Upload your DICOM file</div>
            <div style={{ fontSize: 13, opacity: 0.7 }}>
              Supports NIfTI (.nii, .nii.gz) and DICOM (.dcm) up to 500MB
            </div>
            <input type="file" className="hidden" accept=".dcm,.nii,.nii.gz" />
          </label>

          <div style={{ marginTop: 12, textAlign: "center" }}>
            <label
              style={{
                display: "inline-flex",
                gap: 8,
                alignItems: "center",
                padding: "8px 12px",
                borderRadius: 8,
                border: "1px solid #e5e7eb",
                background: "#fff",
                cursor: "pointer",
              }}
            >
              <input type="file" className="hidden" accept=".dcm,.nii,.nii.gz" />
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
            style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: 24 }}
          >
            <div style={{ fontSize: 18, fontWeight: 600 }}>Try Our Demo</div>
            <p style={{ marginTop: 6, opacity: 0.8 }}>
              Don’t have a DICOM file? Launch the demo viewer and explore the tools.
            </p>
            <div style={{ marginTop: 12 }}>
              <Link
                to="/viewer"
                style={{ background: "#000", color: "#fff", padding: "10px 14px", borderRadius: 8 }}
              >
                ▶︎ Launch Demo Viewer
              </Link>
            </div>
          </div>

          <div
            style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: 24 }}
          >
            <div
              style={{
                height: 160,
                borderRadius: 8,
                border: "1px solid #e5e7eb",
                background: "linear-gradient(135deg,#eef2f7,#e5e7eb)",
                display: "grid",
                placeItems: "center",
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
      <footer style={{ marginTop: 24, background: "#fff", borderTop: "1px solid #e5e7eb" }}>
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
