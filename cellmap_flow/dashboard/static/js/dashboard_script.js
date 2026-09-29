// dashboard_script.js

// The header's "Toggle Dashboard" button. The Neuroglancer column's inline
// flex style makes it fill whatever width the dashboard column leaves, so
// hiding the dashboard is all it takes.
function toggleDashboard() {
  document.getElementById("dashboard-column").classList.toggle("d-none");
}
