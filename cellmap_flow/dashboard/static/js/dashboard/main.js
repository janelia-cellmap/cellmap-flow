// The dashboard page (templates/index.html and the tab partials it
// includes).
import { esc } from "../lib/dom.js";

// The header's "Toggle Dashboard" button. The Neuroglancer column's inline
// flex style makes it fill whatever width the dashboard column leaves, so
// hiding the dashboard is all it takes.
function toggleDashboard() {
  document.getElementById("dashboard-column").classList.toggle("d-none");
}

// For the handlers and inline scripts outside these modules, which call
// these by name: the header's onclick="toggleDashboard()", and esc() for
// scripts escaping text into innerHTML.
window.toggleDashboard = toggleDashboard;
window.esc = esc;
