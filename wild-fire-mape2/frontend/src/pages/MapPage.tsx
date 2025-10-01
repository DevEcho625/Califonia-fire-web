import React, { useEffect, useState } from 'react';
import InteractiveMap from '../components/Map/InteractiveMap';
import Sidebar from '../components/Sidebar';
import { fireService } from '../services/fireService';
import { riskService } from '../services/riskService';
import './MapPage.css';

const MapPage: React.FC = () => {
  const [fires, setFires] = useState([]);
  const [riskData, setRiskData] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState<{lat: number, lng: number} | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [firesData, riskData] = await Promise.all([
        fireService.getCurrentFires(),
        riskService.getRiskData()
      ]);
      setFires(firesData);
      setRiskData(riskData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAreaSelect = async (lat: number, lng: number) => {
    setSelectedLocation({ lat, lng });
    try {
      const risk = await riskService.getRiskForLocation(lat, lng);
      setRiskData(prev => [...prev.filter(r => r.latitude !== lat || r.longitude !== lng), risk]);
    } catch (error) {
      console.error('Error getting risk data:', error);
    }
  };

  return (
    <div className="map-page">
      <Sidebar 
        fires={fires} 
        selectedLocation={selectedLocation}
        onRefresh={loadInitialData}
      />
      <div className="map-container">
        {loading ? (
          <div className="loading">Loading wildfire data...</div>
        ) : (
          <InteractiveMap 
            fires={fires}
            riskData={riskData}
            onAreaSelect={handleAreaSelect}
          />
        )}
      </div>
    </div>
  );
};

export default MapPage;