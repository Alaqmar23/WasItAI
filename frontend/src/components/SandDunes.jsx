import { useRef, useEffect } from 'react';

/**
 * SandDunes — Prismatic Nebula
 * Pixel-stars drifting left→right, viewport-wrapped for uniform density.
 * Scroll creates instant visual parallax offset (no position mutation = no gaps).
 * Mouse creates vortex repulsion with elastic spring-back.
 */
export default function SandDunes() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let w = 0, h = 0;

    const getParticleCount = () => {
      const vw = window.innerWidth;
      if (vw < 640) return 1200;
      if (vw < 1024) return 3000;
      return 7000;
    };

    const PARTICLE_COUNT = getParticleCount();
    const MOUSE_RADIUS = 160;

    let particles = [];
    let mouse = { x: -9999, y: -9999, active: false };

    // Smooth scroll position tracking
    let targetScrollY = window.scrollY;
    let smoothScrollY = window.scrollY;

    const onScroll = () => {
      targetScrollY = window.scrollY;
    };

    // ─── Pixel Class ──────────────────────────────────────────────────────────
    class Pixel {
      constructor() {
        this.init();
      }

      init() {
        this.baseX = Math.random() * w;
        this.baseY = Math.random() * h;
        this.x = this.baseX;
        this.y = this.baseY;
        this.vx = 0;
        this.vy = 0;

        // Drift: left→right with subtle vertical wander
        this.driftX = 0.08 + Math.random() * 0.22;
        this.driftY = -0.01 + Math.random() * 0.02;

        // Depth for parallax (0 = far/slow, 1 = near/fast)
        this.depth = Math.random();
        this.parallax = 0.1 + this.depth * 0.9;

        this.size = 0.4 + Math.random() * 1.6;

        const r = Math.random();
        if (r > 0.82) this.color = '#ffffff';
        else if (r > 0.45) this.color = '#00f7ff';
        else this.color = '#b4a0ff';

        this.alpha = 0.2 + Math.random() * 0.6;
        this.glowFactor = 0;
      }

      update() {
        // Only drift moves the base — scroll never touches baseY
        this.baseX += this.driftX;
        this.baseY += this.driftY;

        // Wrap base around viewport (drift wrapping only)
        if (this.baseX > w) { this.baseX -= w; this.x -= w; }
        if (this.baseX < 0) { this.baseX += w; this.x += w; }
        if (this.baseY > h) { this.baseY -= h; this.y -= h; }
        if (this.baseY < 0) { this.baseY += h; this.y += h; }

        // Mouse vortex repulsion
        // Need visual Y for mouse check
        const visY = this.getVisualY();
        const dx = mouse.x - this.x;
        const dy = mouse.y - visY;
        const distSq = dx * dx + dy * dy;

        this.glowFactor = 0;
        if (mouse.active && distSq < MOUSE_RADIUS * MOUSE_RADIUS) {
          const dist = Math.sqrt(distSq);
          this.glowFactor = Math.pow(1 - dist / MOUSE_RADIUS, 2);

          const force = this.glowFactor;
          const angle = Math.atan2(dy, dx);
          // Radial repulsion
          this.vx -= Math.cos(angle) * force * 0.12;
          this.vy -= Math.sin(angle) * force * 0.12;
          // Tangential swirl → revolving orbit
          this.vx -= Math.sin(angle) * force * 0.18;
          this.vy += Math.cos(angle) * force * 0.18;
        }

        // Spring back toward anchor
        this.x += (this.baseX - this.x) * 0.06;
        this.y += (this.baseY - this.y) * 0.06;

        // Apply and decay velocity
        this.vx *= 0.94;
        this.vy *= 0.94;
        this.x += this.vx;
        this.y += this.vy;
      }

      getVisualY() {
        // Scroll offset applied as visual-only — wraps with modulo so no gaps
        const offset = smoothScrollY * this.parallax * 0.15;
        return ((this.y - offset) % h + h) % h;
      }

      draw() {
        const visY = this.getVisualY();
        const s = this.size + (this.glowFactor * 1.8);
        const a = this.alpha + (this.glowFactor * 0.7);
        ctx.globalAlpha = Math.min(1.0, a);
        ctx.fillStyle = this.glowFactor > 0.3 ? '#ffffff' : this.color;
        ctx.fillRect(this.x, visY, s, s);
      }
    }

    const init = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
      particles = Array.from({ length: PARTICLE_COUNT }, () => new Pixel());
    };

    const animate = () => {
      // Smooth scroll interpolation — instant feel, no jank
      smoothScrollY += (targetScrollY - smoothScrollY) * 0.25;

      // Background color
      const scrollH = document.documentElement.scrollHeight || document.body.scrollHeight || window.innerHeight;
      const scrollRatio = window.scrollY / (scrollH || 1);
      const bgR = Math.floor(7 + scrollRatio * 5);
      const bgG = Math.floor(7 + scrollRatio * 5);
      const bgB = Math.floor(10 + scrollRatio * 20);

      ctx.globalAlpha = 1;
      ctx.fillStyle = `rgb(${bgR}, ${bgG}, ${bgB})`;
      ctx.fillRect(0, 0, w, h);

      for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    // ─── Event Listeners ──────────────────────────────────────────────────────
    const handleMouseMove = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.active = true;
    };
    const handleMouseLeave = () => { mouse.active = false; };
    const handleResize = () => { init(); };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);
    window.addEventListener('resize', handleResize);
    window.addEventListener('scroll', onScroll, { passive: true });

    init();
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('scroll', onScroll);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 0,
        pointerEvents: 'none',
        display: 'block',
      }}
    />
  );
}
