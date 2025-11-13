import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

export default function About() {
  // Load dark mode preference (same as Landing)
  const [isDarkMode] = useState(() => {
    const saved = localStorage.getItem("darkMode");
    return saved === "true";
  });

  useEffect(() => {
    localStorage.setItem("darkMode", isDarkMode.toString());
  }, [isDarkMode]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: isDarkMode ? "#0f172a" : "#f7f7fb",
        color: isDarkMode ? "#f1f5f9" : "#0f172a",
        transition: "all 0.3s ease",
        paddingBottom: 40,
      }}
    >

      {/* top bar */}
      <header
        style={{
          background: isDarkMode ? "#1e293b" : "#fff",
          borderBottom: isDarkMode
            ? "1px solid #334155"
            : "1px solid #e5e7eb",
          height: 80,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          paddingLeft: 32,
          paddingRight: 32,
          fontSize: 28,
          fontWeight: 600,
        }}
      >

        {/* HOME BUTTON */}
        <Link
          to="/"
          style={{
            fontSize: 18,
            fontWeight: 500,
            textDecoration: "none",
            color: isDarkMode ? "#f1f5f9" : "#0f172a",
            opacity: 0.85,
          }}
        >
          ⬅ Home
        </Link>

        <div>About Metis</div>
      </header>

      {/* divider line */}
      <div
        style={{
          width: "100%",
          height: 2,
          background: isDarkMode ? "#334155" : "#d1d5db",
          marginBottom: 40,
        }}
      />

      {/* main content (top row) */}
      <div
        style={{
          maxWidth: 900,
          margin: "0 auto",
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 40,
          padding: "0 24px",
        }}
      >
        {/* TOP LEFT - BROOKS */}
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              width: 120,
              height: 120,
              margin: "0 auto",
              borderRadius: 12,
              border: isDarkMode
                ? "2px solid #334155"
                : "2px solid #cbd5e1",
              background: isDarkMode ? "#1e293b" : "#fff",
              transition: "all 0.3s ease",
            }}
          />
          <p style={{ marginTop: 10, opacity: 0.8 }}>Brooks</p>
        </div>

        {/* TOP MIDDLE - ERICK */}
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              width: 120,
              height: 120,
              margin: "0 auto",
              borderRadius: 12,
              border: isDarkMode
                ? "2px solid #334155"
                : "2px solid #cbd5e1",
              background: isDarkMode ? "#1e293b" : "#fff",
              transition: "all 0.3s ease",
            }}
          />
          <p style={{ marginTop: 10, opacity: 0.8 }}>Erick</p>
        </div>

        {/* TOP RIGHT - JOHN */}
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              width: 120,
              height: 120,
              margin: "0 auto",
              borderRadius: 12,
              border: isDarkMode
                ? "2px solid #334155"
                : "2px solid #cbd5e1",
              background: isDarkMode ? "#1e293b" : "#fff",
              transition: "all 0.3s ease",
            }}
          />
          <p style={{ marginTop: 10, opacity: 0.8 }}>John</p>
        </div>
      </div>

      {/* bottom row */}
      <div
        style={{
          maxWidth: 600,
          margin: "40px auto 0 auto",
          display: "grid",
          gridTemplateColumns: "repeat(2, 1fr)",
          gap: 40,
          padding: "0 24px",
        }}
      >
        {/* BOTTOM LEFT - JACKSON */}
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              width: 120,
              height: 120,
              margin: "0 auto",
              borderRadius: 12,
              border: isDarkMode
                ? "2px solid #334155"
                : "2px solid #cbd5e1",
              background: isDarkMode ? "#1e293b" : "#fff",
              transition: "all 0.3s ease",
            }}
          />
          <p style={{ marginTop: 10, opacity: 0.8 }}>Jackson</p>
        </div>

        {/* BOTTOM RIGHT - TYLER */}
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              width: 120,
              height: 120,
              margin: "0 auto",
              borderRadius: 12,
              border: isDarkMode
                ? "2px solid #334155"
                : "2px solid #cbd5e1",
              background: isDarkMode ? "#1e293b" : "#fff",
              transition: "all 0.3s ease",
            }}
          />
          <p style={{ marginTop: 10, opacity: 0.8 }}>Tyler</p>
        </div>
      </div>
    </div>
  );
}
