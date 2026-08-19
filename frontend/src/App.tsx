import { HashRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Experiments from "./pages/Experiments";
import Incidents from "./pages/Incidents";
import Agents from "./pages/Agents";
import Models from "./pages/Models";

export default function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <nav className="sidebar">
          <div className="brand">
            <div className="brand-name">QI<span>DS</span></div>
            <div className="brand-tagline">Quantum-Assisted Intelligent Detection &amp; Defense System</div>
          </div>
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>Dashboard</NavLink>
          <NavLink to="/experiments" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>Experiments</NavLink>
          <NavLink to="/incidents" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>Incidents</NavLink>
          <NavLink to="/agents" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>Agents</NavLink>
          <NavLink to="/models" className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>Models</NavLink>
          <div className="sidebar-footer">
            <small>Security telemetry from authorized applications, analyzed by classical + quantum-assisted detection.</small>
          </div>
        </nav>
        <main className="main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/incidents" element={<Incidents />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}
