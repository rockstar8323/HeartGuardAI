/**
 * HeartGuard AI — Main JavaScript
 * Smooth animations, form validation, and UI interactions
 */

// ─── Intersection Observer for Scroll Animations ─────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll reveal
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    // Observe cards and sections
    document.querySelectorAll('.card, .metric-card, .chart-card').forEach(el => {
        observer.observe(el);
    });

    // ─── Navbar Scroll Effect ────────────────────────────────────
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            if (currentScroll > 50) {
                navbar.style.background = 'rgba(10, 10, 15, 0.95)';
                navbar.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
            } else {
                navbar.style.background = 'rgba(10, 10, 15, 0.8)';
                navbar.style.boxShadow = 'none';
            }
            lastScroll = currentScroll;
        });
    }

    // ─── Form Validation & UX ────────────────────────────────────
    const form = document.getElementById('predictionForm');
    if (form) {
        // Real-time validation feedback
        form.querySelectorAll('.form-control').forEach(input => {
            input.addEventListener('change', () => {
                if (input.checkValidity()) {
                    input.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                } else {
                    input.style.borderColor = 'rgba(239, 68, 68, 0.5)';
                }
            });

            input.addEventListener('focus', () => {
                input.style.borderColor = 'rgba(168, 85, 247, 0.5)';
            });

            input.addEventListener('blur', () => {
                if (!input.value) {
                    input.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                }
            });
        });

        // Submit with loading state
        form.addEventListener('submit', (e) => {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '⏳ Analyzing...';
                submitBtn.style.opacity = '0.7';
                submitBtn.style.pointerEvents = 'none';
            }
        });
    }

    // ─── Animate Metric Values (Count Up) ────────────────────────
    document.querySelectorAll('.metric-value').forEach(el => {
        const text = el.textContent.trim();
        const match = text.match(/([\d.]+)(%?)/);
        if (!match) return;

        const target = parseFloat(match[1]);
        const suffix = match[2] || '';
        const duration = 1500;
        const start = performance.now();

        el.textContent = '0' + suffix;

        function animate(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = (target * eased).toFixed(1);
            el.textContent = current + suffix;

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                el.textContent = target + suffix;
            }
        }

        requestAnimationFrame(animate);
    });

    // ─── Confidence Bar Animation ────────────────────────────────
    const confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        setTimeout(() => {
            confidenceFill.style.width = confidenceFill.dataset.width;
        }, 500);
    }

    // ─── Smooth Scroll for Anchor Links ──────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ─── Impact Bar Width Animation ──────────────────────────────
    document.querySelectorAll('.impact-bar .bar').forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0px';
        setTimeout(() => {
            bar.style.width = width;
        }, 300);
    });
});
