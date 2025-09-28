// Create the map centered on California
var map = L.map('map').setView([37.5, -120], 6);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
}).addTo(map);

// --- 1. Add some dummy fire locations ---
var fireData = [
  { lat: 38.5, lon: -121.5 }, // Sacramento area
  { lat: 34.2, lon: -118.4 }, // Los Angeles area
  { lat: 36.7, lon: -119.8 },  // Fresno area
    { lat: 39.0, lon: -120.5 } , // Chico area
    { lat: 34.5, lon: -117.3 },  // San Bernardino area
    { lat: 32.7, lon: -117.1 } , // San Diego area
    { lat: 38.0, lon: -122.3 }  // Napa area
];

fireData.forEach(fire => {
  L.circle([fire.lat, fire.lon], {
    radius: 10000, // in meters
    color: "red",
    fillColor: "orange",
    fillOpacity: 0.7
  }).addTo(map).bindPopup("Active Fire");
});

// --- 2. Add a dummy heatmap for risk levels ---
var riskPoints = [
  [37.5, -120, 0.8], // high risk
  [36.5, -121, 0.4], // medium
  [35.0, -119, 0.2]  // low
];

L.heatLayer(riskPoints, {radius: 25}).addTo(map);
