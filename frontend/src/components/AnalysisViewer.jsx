import { useState, useRef, useEffect } from 'react';

const AnalysisViewer = ({ realImage, artifactImage }) => {
  const [isActive, setIsActive] = useState(false);
  const [isTouchDevice, setIsTouchDevice] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    setIsTouchDevice('ontouchstart' in window || navigator.maxTouchPoints > 0);
  }, []);

  const updatePosition = (clientX, clientY) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      containerRef.current.style.setProperty('--mouse-x', `${x}px`);
      containerRef.current.style.setProperty('--mouse-y', `${y}px`);
    }
  };

  // Mouse handlers
  const handleMouseMove = (e) => updatePosition(e.clientX, e.clientY);
  const handleMouseEnter = () => setIsActive(true);
  const handleMouseLeave = () => setIsActive(false);

  // Touch handlers
  const handleTouchStart = (e) => {
    e.preventDefault();
    setIsActive(true);
    const touch = e.touches[0];
    updatePosition(touch.clientX, touch.clientY);
  };
  const handleTouchMove = (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    updatePosition(touch.clientX, touch.clientY);
  };
  const handleTouchEnd = () => setIsActive(false);

  return (
    <div className="analysis-viewer-wrapper glass-panel" style={{ padding: '2rem', borderRadius: '32px', overflow: 'hidden' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h3 className="mono" style={{ color: 'var(--accent-cyan)', fontSize: '1rem', letterSpacing: '2px', marginBottom: '0.5rem', fontWeight: 800 }}>
          AI vs REAL: THE X-RAY TEST
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1rem', lineHeight: 1.5 }}>
          {isTouchDevice
            ? 'Tap and drag over the image to reveal the hidden digital patterns and glitches that AI models leave behind.'
            : 'Move your mouse over the image to reveal the hidden digital patterns and glitches that AI models leave behind.'}
        </p>
      </div>

      <div 
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        className="x-ray-container"
        style={{ 
          position: 'relative', 
          width: '100%', 
          aspectRatio: '16/9', 
          cursor: isTouchDevice ? 'pointer' : 'none', 
          borderRadius: '16px', 
          overflow: 'visible',
          border: '1px solid rgba(255,255,255,0.1)',
          '--mouse-x': '50%',
          '--mouse-y': '50%',
          touchAction: 'none',
        }}
      >
        {/* Base Layer: Authentic Image */}
        <img 
          src={realImage} 
          alt="Authentic reference"
          style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        />

        {/* Top Layer: Artifact Image with Clip-Path */}
        <img 
          src={artifactImage} 
          alt="Neural artifacts"
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'cover', 
            position: 'absolute', 
            top: 0, 
            left: 0,
            clipPath: isActive 
              ? `circle(${isTouchDevice ? '70px' : '100px'} at var(--mouse-x) var(--mouse-y))` 
              : 'circle(0px at 50% 50%)',
            zIndex: 10,
            pointerEvents: 'none',
            transition: isActive ? 'none' : 'clip-path 0.3s ease',
          }}
        />

        {/* The Lens Border & Glow */}
        <div 
          style={{
            position: 'absolute',
            top: 'var(--mouse-y)',
            left: 'var(--mouse-x)',
            width: isTouchDevice ? '140px' : '200px',
            height: isTouchDevice ? '140px' : '200px',
            transform: 'translate(-50%, -50%)',
            border: '2px solid var(--accent-magenta)',
            borderRadius: '50%',
            pointerEvents: 'none',
            zIndex: 20,
            boxShadow: '0 0 40px var(--accent-magenta), inset 0 0 20px var(--accent-magenta)',
            display: isActive ? 'flex' : 'none',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'opacity 0.2s',
            opacity: isActive ? 1 : 0
          }}
        >
          <div className="mono" style={{ fontSize: '0.6rem', color: 'var(--accent-magenta)', background: 'rgba(0,0,0,0.7)', padding: '4px 10px', borderRadius: '4px', fontWeight: 700 }}>
            ARTIFACTS_DETECTED
          </div>
        </div>

        {/* Scan Bar */}
        <div 
          style={{
            position: 'absolute',
            top: 'var(--mouse-y)',
            left: 'var(--mouse-x)',
            width: isTouchDevice ? '120px' : '180px',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            zIndex: 25,
            display: isActive ? 'block' : 'none',
          }}
        >
          <div 
            className="neural-scan-bar"
            style={{
              width: '100%',
              height: '2px',
              background: 'var(--accent-magenta)',
              boxShadow: '0 0 15px var(--accent-magenta)',
              animation: 'vibe-scan 2s ease-in-out infinite'
            }}
          />
        </div>

        {/* Mobile tap hint overlay */}
        {isTouchDevice && !isActive && (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 15,
            pointerEvents: 'none',
          }}>
            <div className="mono" style={{
              background: 'rgba(0,0,0,0.6)',
              padding: '0.6rem 1.2rem',
              borderRadius: '30px',
              fontSize: '0.7rem',
              color: 'var(--accent-cyan)',
              border: '1px solid rgba(0,229,255,0.2)',
              letterSpacing: '1px',
              animation: 'fadeInUp 1s ease forwards',
            }}>
              TAP &amp; HOLD TO SCAN
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisViewer;

