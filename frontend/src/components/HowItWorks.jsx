import AnalysisViewer from './AnalysisViewer';

export default function HowItWorks({ className }) {
  return (
    <div className={`how-it-works-section ${className || ''}`} style={{ padding: '0 5% 4rem 5%', display: 'flex', flexDirection: 'column', gap: '3rem', gridColumn: '1 / -1', position: 'relative', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
      <div style={{ position: 'absolute', top: '10%', right: '10%', width: 300, height: 300, background: 'var(--glow-magenta)', filter: 'blur(100px)', opacity: 0.4, zIndex: -1, borderRadius: '50%' }} />

      <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
        <h2 style={{ fontSize: '2.2rem', fontWeight: 800, letterSpacing: '-0.5px', marginBottom: '0.8rem' }}>How Information is Processed</h2>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto', fontSize: '1rem' }}>
          WasItAI analyzes latent noise patterns and frequency artifacts that are biologically impossible for authentic photography.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.9fr', gap: '5rem', alignItems: 'center' }}>

        {/* Sequence Steps */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', padding: '1.5rem', borderRadius: '20px', transition: 'all 0.3s ease', cursor: 'default', background: 'rgba(255,255,255,0.01)' }} onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; e.currentTarget.style.transform = 'translateX(10px)' }} onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.01)'; e.currentTarget.style.transform = 'translateX(0)' }}>
            <div style={{ width: '45px', height: '45px', borderRadius: '50%', background: 'rgba(0, 229, 255, 0.1)', border: '1px solid var(--accent-cyan)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)', fontWeight: 800, flexShrink: 0, boxShadow: '0 0 20px var(--glow-cyan)' }}>1</div>
            <div>
              <h4 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.5rem', color: '#fff' }}>Input Evidence</h4>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>Upload raw imagery into the secure neural receptor zone for high-fidelity scanning.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', padding: '1.5rem', borderRadius: '20px', transition: 'all 0.3s ease', cursor: 'default', background: 'rgba(255,255,255,0.01)' }} onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; e.currentTarget.style.transform = 'translateX(10px)' }} onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.01)'; e.currentTarget.style.transform = 'translateX(0)' }}>
            <div style={{ width: '45px', height: '45px', borderRadius: '50%', background: 'rgba(0, 229, 255, 0.1)', border: '1px solid var(--accent-cyan)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)', fontWeight: 800, flexShrink: 0, boxShadow: '0 0 20px var(--glow-cyan)' }}>2</div>
            <div>
              <h4 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.5rem', color: '#fff' }}>Neural Extraction</h4>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>The EfficientNet-V2-S core identifies millions of deep features, looking for mathematical inconsistency.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', padding: '1.5rem', borderRadius: '20px', transition: 'all 0.3s ease', cursor: 'default', background: 'rgba(255,255,255,0.01)' }} onMouseOver={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; e.currentTarget.style.transform = 'translateX(10px)' }} onMouseOut={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.01)'; e.currentTarget.style.transform = 'translateX(0)' }}>
            <div style={{ width: '45px', height: '45px', borderRadius: '50%', background: 'rgba(255, 0, 85, 0.1)', border: '1px solid var(--accent-magenta)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-magenta)', fontWeight: 800, flexShrink: 0, boxShadow: '0 0 20px var(--glow-magenta)' }}>3</div>
            <div>
              <h4 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.5rem', color: '#fff' }}>Forensic Verdict</h4>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>Receive an absolute classification based on weighted neural probability maps.</p>
            </div>
          </div>

        </div>

        {/* Interactive Neural X-Ray Section */}
        <div className="fade-in-section">
          <AnalysisViewer
            realImage="/authentic.png"
            artifactImage="/artifacts.png"
            description="Comparative Analysis Prototype: Authentic photography vs. AI-Generated Latent Noise. Hover to manually inspect."
          />
        </div>

      </div>
    </div>
  );
}
