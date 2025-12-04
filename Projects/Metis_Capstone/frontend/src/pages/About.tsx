import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

const teamMembers = [
  {
    id: "brooks",
    name: "Brooks",
    role: "Team Member",
    bio: "Brooks Schafer is from Baton Rouge and has attended Southeastern since the fall of 2019. He is graduating in December 2025 with a degree in Computer Science and a concentration in Data Science. Once graduating, he plans to pursue a career in data analytics. Outside of computer science, he enjoys playing music and video games.",
    image: "/images/Brooks.png"
  },
  {
    id: "erick",
    name: "Erick",
    role: "Team Member",
    bio: "I'm Erick, group leader of Metis. I began my journey at SELU in 2022 and I'll be leaving December 2025. Since I began, I've been developing my programming skills and recently I fell in love with neuroscience. Specifically neuroimaging, hence the creation of this project. I hope to continue my work in the medical field one day!",
    image: "/images/Erick.jpg"
  },
  {
    id: "john",
    name: "John",
    role: "Team Member",
    bio: "John Montz has attended Southeastern Louisiana University since the Summer of 2024 and plans to graduate with a degree in Information Technology in 2026. John has been working for Geek Squad as an ARA since 2023. Outside of his coursework, John actively develops and maintains a personal home lab to strengthen his cybersecurity skills and researches current and emerging trends in AI and software development. In his spare time, John enjoys playing video games, coding cybersecurity tools and scripts, and experimenting with local AI systems.",
    image: "/images/John.jpg"
  },
  {
    id: "jackson",
    name: "Jackson",
    role: "Team Member",
    bio: "Jackson Eason has attended Southeastern since the fall of 2021 and plans to graduate with a degree in Computer Science in 2025. Jackson has worked an internship at Laborde Products in Covington, LA for the past 6 months. When not studying computer science, Jackson likes to play golf, play video games, and watch the Saints.",
    image: "/images/Jackson.png"
  },
  {
    id: "tyler",
    name: "Tyler",
    role: "Team Member",
    bio: "I am Tyler Leggio. I am a student at SELU graduating in fall 2025 and I am the frontend developer for Metis DICOM. Originally from Baton Rouge and moved to Hammond in 2017 to be close to school. After I graduate I am hoping to be a Network Administrator and hopefully a Video Game developer in the future.",
    image: "/images/Tyler.png"
  }
];


type TeamMember = {
  id: string;
  name: string;
  role: string;
  bio: string;
  image: string;
};

const TeamCard = ({ member, isDarkMode }: { member: TeamMember; isDarkMode: boolean }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        background: isDarkMode ? "#1e293b" : "#ffffff",
        borderRadius: "16px",
        padding: "24px",
        textAlign: "center",
        boxShadow: isHovered
          ? (isDarkMode ? "0 10px 25px -5px rgba(0, 0, 0, 0.5)" : "0 10px 25px -5px rgba(0, 0, 0, 0.1)")
          : (isDarkMode ? "0 4px 6px -1px rgba(0, 0, 0, 0.3)" : "0 4px 6px -1px rgba(0, 0, 0, 0.05)"),
        transform: isHovered ? "translateY(-5px)" : "translateY(0)",
        transition: "all 0.3s ease",
        border: isDarkMode ? "1px solid #334155" : "1px solid #e2e8f0",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center"
      }}
    >
      <div
        style={{
          width: 120,
          height: 120,
          marginBottom: 16,
          borderRadius: "50%",
          overflow: "hidden",
          border: isDarkMode ? "3px solid #64748b" : "3px solid #e2e8f0",
          background: "#ccc"
        }}
      >
        {member.image ? (
          <img
            src={member.image}
            alt={member.name}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 40, color: "#555" }}>
            {member.name[0]}
          </div>
        )}
      </div>

      <h3 style={{ margin: "0 0 4px 0", fontSize: "1.25rem", color: isDarkMode ? "#f1f5f9" : "#0f172a" }}>
        {member.name}
      </h3>

      <span style={{ fontSize: "0.85rem", color: isDarkMode ? "#94a3b8" : "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12 }}>
        {member.role}
      </span>

      <p style={{ fontSize: "0.95rem", lineHeight: "1.5", color: isDarkMode ? "#cbd5e1" : "#475569", margin: 0 }}>
        {member.bio}
      </p>
    </div>
  );
};

export default function About() {
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
        paddingBottom: 60,
      }}
    >
      <header
        style={{
          background: isDarkMode ? "#1e293b" : "#ffffff",
          borderBottom: isDarkMode ? "1px solid #334155" : "1px solid #e5e7eb",
          height: 80,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          paddingLeft: 32,
          paddingRight: 32,
          boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
          position: "sticky",
          top: 0,
          zIndex: 10
        }}
      >
        <Link
          to="/"
          style={{
            fontSize: 18,
            fontWeight: 500,
            textDecoration: "none",
            color: isDarkMode ? "#f1f5f9" : "#0f172a",
            display: "flex",
            alignItems: "center",
            gap: "8px"
          }}
        >
          <span>←</span> Home
        </Link>

        <div style={{ fontSize: 24, fontWeight: 700 }}>About <span style={{color: "#6366f1"}}>Metis</span></div>
      </header>

      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "40px 24px" }}>

        <div style={{ textAlign: "center", marginBottom: 50 }}>
          <h1 style={{ fontSize: "2.5rem", marginBottom: "16px" }}>Meet the Team</h1>
          <p style={{ fontSize: "1.1rem", color: isDarkMode ? "#94a3b8" : "#64748b", maxWidth: 600, margin: "0 auto" }}>
            The developers and designers behind the Metis project.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 30,
            marginBottom: 30,
          }}
        >
          {teamMembers.slice(0, 3).map((member) => (
            <TeamCard key={member.id} member={member} isDarkMode={isDarkMode} />
          ))}
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 30,
            maxWidth: 700,
            margin: "0 auto"
          }}
        >
          {teamMembers.slice(3, 5).map((member) => (
            <TeamCard key={member.id} member={member} isDarkMode={isDarkMode} />
          ))}
        </div>

      </div>
    </div>
  );
}
