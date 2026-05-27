import React, { useState, useRef, useEffect } from 'react';
import SandDunes from './components/SandDunes';
import HowItWorks from './components/HowItWorks';
import Footer from './components/Footer';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [animatedConfidence, setAnimatedConfidence] = useState(0);
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const [lastFeedbackCorrect, setLastFeedbackCorrect] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (result) {
      setAnimatedConfidence(0);
      const duration = 2000; // 2 seconds
      const steps = 60;
      const increment = result.confidence / steps;
      let current = 0;
      const timer = setInterval(() => {
        current += increment;
        if (current >= result.confidence) {
          setAnimatedConfidence(result.confidence.toFixed(2));
          clearInterval(timer);
        } else {
          setAnimatedConfidence(current.toFixed(2));
        }
      }, duration / steps);
      return () => clearInterval(timer);
    }
  }, [result]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;

    const startTime = Date.now();
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Force a minimum scan time of 2.5s to allow animation to complete
      const [response] = await Promise.all([
        fetch('https://alaqmar-wasitai-backend.hf.space/predict', {
          method: 'POST',
          body: formData,
        }),
        new Promise(resolve => setTimeout(resolve, 2500))
      ]);

      if (!response.ok) throw new Error('Neural response failed');

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Neural scan interrupted. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (isCorrect) => {
    setFeedbackGiven(true);
    setLastFeedbackCorrect(isCorrect);
    try {
      const formData = new FormData();
      formData.append('is_correct', isCorrect);
      formData.append('prediction', result.prediction);
      formData.append('image_id', result.image_id);

      // If incorrect, send the original file for verified storage
      if (!isCorrect && file) {
        formData.append('file', file);
      }

      await fetch('https://alaqmar-wasitai-backend.hf.space/feedback', {
        method: 'POST',
        body: formData,
      });
    } catch (err) {
      console.error('Failed to submit feedback', err);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setFeedbackGiven(false);
    setLastFeedbackCorrect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="app-container">
      {/* Background Component */}
      <SandDunes />

      {/* Header */}
      <header className="header-main">
        <div className="logo-section" onClick={() => window.location.reload()}>
          <div className="logo-outer-wrapper">
            <div className="logo-inner-container">
              <svg viewBox="0 0 100 100" style={{ width: '100%', height: '100%' }}>
                <defs>
                  <linearGradient id="shell-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#4a4a4f" />
                    <stop offset="50%" stopColor="#2a2a2f" />
                    <stop offset="100%" stopColor="#4a4a4f" />
                  </linearGradient>
                  <filter id="shell-depth">
                    <feDropShadow dx="-1" dy="1" stdDeviation="1" floodOpacity="0.4" />
                  </filter>
                  <mask id="logo-mask-header">
                    <rect x="0" y="0" width="100" height="100" fill="white" />
                    <rect x="44" y="0" width="12" height="100" fill="black" />
                  </mask>
                </defs>
                {/* Wrap in a group for better mobile mask support */}
                <g mask="url(#logo-mask-header)">
                  <ellipse cx="50" cy="50" rx="39.16" ry="35" fill="none" stroke="url(#shell-gradient)" strokeWidth="7" style={{ filter: 'url(#shell-depth)' }} />
                </g>
              </svg>
            </div>
            {/* Refined Length Dynamic Beams */}
            <div className="logo-beam-left" style={{ position: 'absolute', top: '9%', left: '-5px', width: '2px', height: '82%', background: 'var(--accent-cyan)', boxShadow: '0 0 10px var(--accent-cyan)', zIndex: 10 }}></div>
            <div className="logo-beam-right" style={{ position: 'absolute', top: '9%', right: '-5px', width: '2px', height: '82%', background: 'var(--accent-cyan)', boxShadow: '0 0 10px var(--accent-cyan)', zIndex: 10 }}></div>
          </div>
          <span className="logo-text">WasIt<span className="logo-ai">AI</span></span>
        </div>

        <nav className="nav-container">
          <a href="#" className="nav-link active">IMAGE SCAN</a>
          <a href="#" className="nav-link disabled">VIDEO SCAN <span className="badge">LOCKED</span></a>
        </nav>
      </header>

      {/* Main Content Area Using index.css Grid */}
      <main className="main-grid">

        {/* Left Column: Image Upload / Analysis */}
        <div className="main-content fade-in-section">
          <section className="hero-section">
            <div className="hero-glow"></div>
            <h1 className="hero-title">Was It AI?</h1>
            <h2 className="hero-subtitle">Neural Image Analysis.</h2>
          </section>

          <div className="detector-card glass-panel upload-hover-card">
            {/* Upload State */}
            {!preview && !error && !result && (
              <div
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
              >
                <input ref={fileInputRef} type="file" hidden onChange={handleFileChange} accept="image/*" />
                <div style={{
                  width: '50px',
                  height: '50px',
                  borderRadius: '50%',
                  background: 'rgba(255,255,255,0.05)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1rem',
                  border: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: '#fff' }}>
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                </div>
                <h3 className="mono" style={{ letterSpacing: '1px', marginBottom: '0.3rem', fontSize: '1rem', fontWeight: 600 }}>UPLOAD OR DROP IMAGE</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>High resolution JPG, PNG or WebP</p>
              </div>
            )}

            {/* Preview State - Before Scan */}
            {preview && !loading && !result && !error && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ flex: 1, position: 'relative', borderRadius: '12px', overflow: 'hidden', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)' }}>
                  <img src={preview} alt="Upload" style={{ width: '100%', height: '300px', objectFit: 'contain', zIndex: 2 }} />
                </div>

                <div style={{ display: 'flex', gap: '1rem' }}>
                  <button
                    onClick={reset}
                    className="mono"
                    style={{
                      padding: '1rem',
                      background: 'rgba(255,255,255,0.05)',
                      color: 'var(--text-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: 600,
                      transition: 'all 0.3s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                    onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                  >
                    DISCARD
                  </button>
                  <button
                    onClick={handlePredict}
                    className="mono scan-btn"
                    style={{
                      flex: 1,
                      padding: '1rem',
                      background: 'var(--accent-cyan)',
                      color: '#000',
                      border: 'none',
                      borderRadius: '8px',
                      fontWeight: 'bold',
                      letterSpacing: '1px',
                      cursor: 'pointer',
                      fontSize: '1rem'
                    }}
                  >
                    LAUNCH NEURAL SCAN
                  </button>
                </div>
              </div>
            )}

            {/* Loading State - X-Ray Scanning Line */}
            {loading && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden', borderRadius: '12px', background: 'rgba(0,0,0,0.6)', padding: '2rem' }}>
                {/* The Image being scanned */}
                <img src={preview} alt="Scanning" style={{ maxWidth: '90%', maxHeight: '80%', objectFit: 'contain', opacity: 0.6, borderRadius: '8px', zIndex: 1 }} />

                {/* The X-Ray Scanning Strip */}
                <div className="scan-overlay"></div>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{
                  border: '1px solid var(--accent-magenta)',
                  background: 'rgba(255, 0, 85, 0.1)',
                  borderRadius: '12px',
                  padding: '2rem',
                  width: '100%',
                  textAlign: 'center'
                }}>
                  <h3 className="mono" style={{ color: '#fff', marginBottom: '1rem', fontSize: '1.2rem', fontWeight: 'bold' }}>ERROR</h3>
                  <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>{error}</p>
                  <button
                    onClick={reset}
                    className="mono"
                    style={{
                      padding: '0.8rem 2rem',
                      background: 'transparent',
                      border: '1px solid rgba(255,255,255,0.2)',
                      color: '#fff',
                      borderRadius: '8px',
                      cursor: 'pointer'
                    }}
                  >
                    TRY AGAIN
                  </button>
                </div>
              </div>
            )}

            {/* Result State */}
            {result && !loading && !error && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', overflow: 'hidden' }}>
                  <img src={preview} style={{ width: '100%', height: '300px', objectFit: 'contain', opacity: 0.8 }} />

                </div>

                <div className="result-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1.5rem' }}>
                  <div>
                    <p className="mono" style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', letterSpacing: '2px', marginBottom: '5px' }}>{result.prediction === 'AI' ? 'AI GENERATED' : 'AUTHENTIC'}</p>
                    <h2 style={{ fontSize: '2.5rem', color: '#fff', margin: 0, fontWeight: 800 }}>{animatedConfidence}%</h2>
                  </div>
                  <button
                    onClick={reset}
                    className="mono scan-another-btn"
                    style={{
                      background: 'transparent',
                      border: '1px solid rgba(255,255,255,0.2)',
                      padding: '10px 20px',
                      color: '#fff',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      transition: 'all 0.3s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                    onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    SCAN ANOTHER
                  </button>
                </div>

                <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${animatedConfidence}%`,
                    height: '100%',
                    background: result.prediction === 'AI' ? 'var(--accent-magenta)' : 'var(--accent-cyan)',
                    transition: 'width 1s cubic-bezier(0.16, 1, 0.3, 1)'
                  }}></div>
                </div>

                {/* Feedback Section */}
                <div className="fade-in-section" style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.2rem', padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  {!feedbackGiven ? (
                    <>
                      <p className="mono" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>Was this prediction correct?</p>
                      <div style={{ display: 'flex', gap: '1rem', width: '100%' }}>
                        <button onClick={() => handleFeedback(true)} style={{ flex: 1, padding: '0.6rem', background: 'rgba(0, 229, 255, 0.1)', border: '1px solid var(--accent-cyan)', color: 'var(--accent-cyan)', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, transition: 'all 0.2s' }}>YES</button>
                        <button onClick={() => handleFeedback(false)} style={{ flex: 1, padding: '0.6rem', background: 'rgba(255, 0, 85, 0.1)', border: '1px solid var(--accent-magenta)', color: 'var(--accent-magenta)', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, transition: 'all 0.2s' }}>NO</button>
                      </div>
                    </>
                  ) : (
                    <p className="mono pulse-glow-text" style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', margin: 0 }}>
                      {lastFeedbackCorrect ? '✓ Verification complete' : '✓ Image saved for evaluation'}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="video-section-container fade-in-section">
          <div className="video-detection-card glass-panel upload-hover-card">
            <div className="video-card-overlay"></div>

            <div className="video-info-container">
              <div className="video-header">
                <span className="video-title">Sequence Scan</span>
                <span className="badge-video mono" style={{ background: 'rgba(255, 255, 255, 0.05)', letterSpacing: '1px' }}>LOCKED FEATURE</span>
              </div>

              <p className="video-desc" style={{ fontSize: '1.05rem', lineHeight: '1.7', marginTop: '1rem' }}>
                Advanced temporal frame analysis and high-framerate multi-modal sequence decoding are currently undergoing supervised tuning.
              </p>

              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
                <div className="coming-soon-box" style={{
                  padding: '1.5rem 3.5rem',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                  background: 'rgba(255, 255, 255, 0.03)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backdropFilter: 'blur(10px)',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
                }}>
                  <span className="mono pulse-glow-text" style={{
                    color: 'var(--accent-cyan)',
                    letterSpacing: '4px',
                    fontWeight: '800',
                    fontSize: '0.9rem'
                  }}>COMING SOON</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <section id="how-it-works">
        <HowItWorks />
      </section>

      <Footer />

      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes spin {
            100% { transform: rotate(360deg); }
        }
        @keyframes pulseRing {
            0% { transform: translate(-50%, -50%) scale(0.95); box-shadow: 0 0 0 0 rgba(0, 229, 255, 0.7); }
            70% { transform: translate(-50%, -50%) scale(1); box-shadow: 0 0 0 20px rgba(0, 229, 255, 0); }
            100% { transform: translate(-50%, -50%) scale(0.95); box-shadow: 0 0 0 0 rgba(0, 229, 255, 0); }
        }
        @keyframes beamScanLeft {
            0% { left: -100%; opacity: 0; }
            50% { left: 45%; opacity: 1; width: 5px; box-shadow: 0 0 20px 2px var(--accent-cyan); }
            100% { left: 45%; opacity: 0; }
        }
        @keyframes beamScanRight {
            0% { right: -100%; opacity: 0; }
            50% { right: 45%; opacity: 1; width: 5px; box-shadow: 0 0 20px 2px var(--accent-cyan); }
            100% { right: 45%; opacity: 0; }
        }
        
        .beam-scan-left {
            position: absolute;
            top: 0;
            height: 100%;
            width: 20px;
            background: linear-gradient(90deg, transparent, var(--accent-cyan));
            animation: beamScanLeft 2s ease-in-out infinite;
            pointer-events: none;
            z-index: 5;
        }
        .beam-scan-right {
            position: absolute;
            top: 0;
            height: 100%;
            width: 20px;
            background: linear-gradient(-90deg, transparent, var(--accent-cyan));
            animation: beamScanRight 2s ease-in-out infinite;
            pointer-events: none;
            z-index: 5;
        }

        .pulse-glow {
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
            animation: glowPulse 2s infinite;
        }
        @keyframes glowPulse {
            0% { box-shadow: 0 0 15px rgba(0, 229, 255, 0.3); }
            50% { box-shadow: 0 0 30px rgba(0, 229, 255, 0.6); }
            100% { box-shadow: 0 0 15px rgba(0, 229, 255, 0.3); }
        }
        .pulse-glow-text {
            animation: textPulse 2s infinite;
        }
        @keyframes textPulse {
            0% { text-shadow: 0 0 10px rgba(0,229,255,0.4); }
            50% { text-shadow: 0 0 20px rgba(0,229,255,0.8); }
            100% { text-shadow: 0 0 10px rgba(0,229,255,0.4); }
        }

        .upload-dropzone:hover {
            background: rgba(0, 229, 255, 0.03) !important;
            border-color: var(--accent-cyan) !important;
            transform: translateY(-2px);
        }

        .upload-hover-card {
            transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s ease;
        }
        .upload-hover-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.8), inset 0 0 30px rgba(255,255,255,0.06);
        }
        
        .scan-btn:hover {
            box-shadow: 0 0 30px rgba(0,229,255,0.4);
            transform: scale(1.02);
        }
      `}} />
    </div>
  );
}
