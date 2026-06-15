import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
    const [history, setHistory] = useState([
        { date: '2023-01', risk: 0.2 },
        { date: '2023-05', risk: 0.35 },
        { date: '2024-01', risk: 0.75 }, // Showing progression
    ]);

    return (
        <div className="dashboard-container" style={{ display: 'flex', gap: '20px', padding: '30px' }}>
            {/* Sidebar: Patient Info */}
            <div style={{ width: '300px', backgroundColor: '#f4f7f6', padding: '20px', borderRadius: '10px' }}>
                <h2>Patient Profile</h2>
                <p><strong>Name:</strong> John Doe</p>
                <p><strong>ID:</strong> P-9921</p>
                <hr />
                <h3>Recent Action</h3>
                <button style={{ width: '100%', padding: '10px', backgroundColor: '#007bff', color: '#fff', border: 'none', borderRadius: '5px' }}>
                    Upload New Scan
                </button>
            </div>

            {/* Main Content: Analytics */}
            <div style={{ flex: 1 }}>
                <h1>Disease Progression Trend</h1>
                <div style={{ height: '300px', width: '100%', backgroundColor: '#fff', padding: '10px', borderRadius: '10px', boxShadow: '0 4px 8px rgba(0,0,0,0.1)' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="risk" stroke="#ff4d4f" strokeWidth={3} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    <div style={{ padding: '20px', backgroundColor: '#e6f7ff', borderRadius: '10px' }}>
                        <h3>Hybrid Risk Score</h3>
                        <h1 style={{ color: '#0050b3' }}>82%</h1>
                        <p>High Priority Case</p>
                    </div>
                    <div style={{ padding: '20px', backgroundColor: '#f6ffed', borderRadius: '10px' }}>
                        <h3>AI Recommendation</h3>
                        <p>Perform FibroScan immediately. Limit sodium intake.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;