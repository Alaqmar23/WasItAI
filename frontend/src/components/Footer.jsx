import React from 'react';

const Footer = () => {
  return (
    <footer className="footer-container fade-in-section" style={{ marginTop: '8rem', paddingBottom: '4rem', padding: '0 5%' }}>
      <div className="glass-panel" style={{ padding: '3rem', borderTop: '1px solid var(--border-color)', maxWidth: '1200px', margin: '0 auto' }}>

        <div className="footer-grid" style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.9fr', gap: '5rem' }}>

          {/* About Section */}
          <div className="about-section">
            <h3 className="mono" style={{ color: 'var(--accent-cyan)', marginBottom: '1.2rem', fontSize: '0.85rem', letterSpacing: '4px' }}>
              ABOUT THE PLATFORM
            </h3>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.7', fontSize: '0.95rem', fontWeight: 300 }}>
              WasItAI is a state-of-the-art AI detection platform designed to protect digital integrity.
              By leveraging deep learning architectures like EfficientNet-V2, we provide specialized
              analysis to distinguish between authentic photography and AI-generated imagery.
            </p>
          </div>

          {/* Contact Section */}
          <div className="contact-section">
            <h3 className="mono" style={{ color: 'var(--accent-cyan)', marginBottom: '1.2rem', fontSize: '0.85rem', letterSpacing: '4px' }}>
              REACH OUT
            </h3>
            <div className="contact-links" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <a href="https://github.com/Alaqmar23" target="_blank" rel="noopener noreferrer" className="nav-link" style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '0.9rem' }}>
                <span className="mono" style={{ color: 'var(--accent-cyan)' }}>GITHUB //</span> @Alaqmar23
              </a>
              <a href="mailto:wasitai.web@gmail.com" className="nav-link" style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '0.9rem' }}>
                <span className="mono" style={{ color: 'var(--accent-cyan)' }}>ENQUIRY //</span> wasitai.web@gmail.com
              </a>
              <a href="mailto:alaqmarkanchwala4@gmail.com" className="nav-link" style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '0.9rem' }}>
                <span className="mono" style={{ color: 'var(--accent-cyan)' }}>PERSONAL //</span> alaqmarkanchwala4@gmail.com
              </a>
            </div>
          </div>

        </div>

        <div className="footer-bottom mono" style={{ marginTop: '3rem', paddingTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)', textAlign: 'center', opacity: 0.4, fontSize: '0.65rem', letterSpacing: '2px' }}>
          &copy; 2026 WASITAI NEURAL SYSTEMS // PROTECTING PHOTOGRAPHIC INTEGRITY
        </div>
      </div>
    </footer>
  );
};

export default Footer;
