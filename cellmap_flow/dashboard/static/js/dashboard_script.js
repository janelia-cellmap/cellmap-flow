// dashboard_script.js

function toggleDashboard() {
  const dashboardCol = document.getElementById("dashboard-column");
  const iframeCol = document.getElementById("iframe-column");

  // If dashboard is currently hidden
  if (dashboardCol.classList.contains("d-none")) {
    // Show the dashboard
    dashboardCol.classList.remove("d-none");
    dashboardCol.classList.add("col-3");
    
    // Shrink iframe column to col-9
    iframeCol.classList.remove("col-12");
    iframeCol.classList.add("col-9");
  } else {
    // Hide the dashboard
    dashboardCol.classList.add("d-none");
    dashboardCol.classList.remove("col-3");
    
    // Expand iframe column to col-12
    iframeCol.classList.remove("col-9");
    iframeCol.classList.add("col-12");
  }
}

// // Wait until DOM is loaded
// document.addEventListener('DOMContentLoaded', function() {
//     // Get references to checkboxes
//     const chkSimple = document.getElementById('methodSimple');
//     const chkContrast = document.getElementById('methodContrast');
//     const chkStandardization = document.getElementById('methodStandardization');
  
//     // Parameter sections
//     const simpleParams = document.getElementById('simpleParams');
//     const contrastParams = document.getElementById('contrastParams');
//     const standardizationParams = document.getElementById('standardizationParams');
  
//     // Show/hide parameter sections based on checkbox
//     function toggleVisibility(checkbox, paramsDiv) {
//       if (checkbox.checked) {
//         paramsDiv.style.display = 'block';
//       } else {
//         paramsDiv.style.display = 'none';
//       }
//     }
  
//     // Event listeners for each checkbox
//     chkSimple.addEventListener('change', () => toggleVisibility(chkSimple, simpleParams));
//     chkContrast.addEventListener('change', () => toggleVisibility(chkContrast, contrastParams));
//     chkStandardization.addEventListener('change', () => toggleVisibility(chkStandardization, standardizationParams));
  
//     // Initialize display state
//     toggleVisibility(chkSimple, simpleParams);
//     toggleVisibility(chkContrast, contrastParams);
//     toggleVisibility(chkStandardization, standardizationParams);
  
//     // Handle Input form submission
//     document.getElementById('submitInput').addEventListener('click', function() {
//       const data = {
//         "norm_type": "min_max",
//         "min_value": 0,
//         "max_value": 1
//       };
//       console.log(data);
  
//       // If you rely on Jinja templating for "{{ inference_servers }}", ensure this .js file is also
//       // processed by Jinja or pass the variable via a global script variable in the HTML.
//       fetch("{{ inference_servers }}/input_normalize", {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(data),
//       })
//       .then(response => response.json())
//       .then(responseData => {
//         console.log(responseData);
//         alert("Data submitted (Input): " + JSON.stringify(responseData));
//       })
//       .catch(error => {
//         console.error(error);
//       });
  
//       var iframe = document.getElementById('my_iframe');
//       iframe.src = iframe.src;
//       console.log('Iframe refreshed');
//     });
  
//     // Handle Output form submission
//     document.getElementById('submitOutput').addEventListener('click', function() {
//       const data = {
//         outputChannel: document.getElementById('outputChannel').value
//       };
  
//       fetch('/api/process', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(data),
//       })
//       .then(response => response.json())
//       .then(responseData => {
//         console.log(responseData);
//         alert("Data submitted (Output): " + JSON.stringify(responseData));
//       })
//       .catch(error => {
//         console.error(error);
//       });
//     });
//   });
  