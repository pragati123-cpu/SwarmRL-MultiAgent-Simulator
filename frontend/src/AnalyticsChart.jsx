import React from 'react';

export function AnalyticsChart({ data = [] }) {
  const maxExplored = 100;
  
  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      width: '320px',
      backgroundColor: 'rgba(15, 23, 42, 0.85)',
      backdropFilter: 'blur(8px)',
      border: '1px solid rgba(56, 189, 248, 0.3)',
      borderRadius: '12px',
      padding: '16px',
      color: '#fff',
      boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      zIndex: 100
    }}>
      <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '12px', color: '#38bdf8', display: 'flex', justifyContent: 'space-between' }}>
        <span>Map Exploration Progress</span>
        <span style={{ color: '#4ade80' }}>
          {data.length > 0 ? `${data[data.length - 1].explored}%` : '0%'}
        </span>
      </div>

      {/* SVG Real-time Line Chart */}
      <div style={{ height: '100px', width: '100%', position: 'relative' }}>
        <svg width="100%" height="100%" viewBox="0 0 300 100" style={{ overflow: 'visible' }}>
          {/* Background Grid Lines */}
          <line x1="0" y1="20" x2="300" y2="20" stroke="#334155" strokeDasharray="4 4" />
          <line x1="0" y1="50" x2="300" y2="50" stroke="#334155" strokeDasharray="4 4" />
          <line x1="0" y1="80" x2="300" y2="80" stroke="#334155" strokeDasharray="4 4" />

          {/* Polyline Data Rendering */}
          {data.length > 1 && (
            <polyline
              fill="none"
              stroke="#38bdf8"
              strokeWidth="3"
              points={data.map((item, index) => {
                const x = (index / (data.length - 1)) * 300;
                const y = 100 - (item.explored / maxExplored) * 90;
                return `${x},${y}`;
              }).join(' ')}
            />
          )}

          {/* Data Points */}
          {data.map((item, index) => {
            const x = (index / (data.length - 1 || 1)) * 300;
            const y = 100 - (item.explored / maxExplored) * 90;
            return (
              <circle key={index} cx={x} cy={y} r="4" fill="#00d9ff" stroke="#ffffff" strokeWidth="1.5" />
            );
          })}
        </svg>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginTop: '8px' }}>
        <span>0s</span>
        <span>25s</span>
      </div>
    </div>
  );
}