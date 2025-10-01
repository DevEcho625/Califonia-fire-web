import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Provider } from 'react-redux';
import { store } from './store';
import MapPage from './pages/MapPage';
import SafetyPage from './pages/SafetyPage';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  return (
    <Provider store={store}>
      <Router>
        <div className="App">
          <Navbar />
          <Routes>
            <Route path="/" element={<MapPage />} />
            <Route path="/safety" element={<SafetyPage />} />
          </Routes>
        </div>
      </Router>
    </Provider>
  );
}

export default App;