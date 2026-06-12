document.addEventListener('copy', function(e) {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed) return;

  const range = selection.getRangeAt(0);

  // Every <math> in document order. Bail unless the selection touches one.
  const liveMath = Array.from(document.querySelectorAll('math'));
  const touched = liveMath.filter(m => range.intersectsNode(m));
  if (touched.length === 0) return;

  // Pull the full LaTeX source out of a live <math>'s TeX annotation.
  const texOf = math => {
    const annotation = math.querySelector('annotation[encoding="application/x-tex"]');
    return annotation ? '$$' + annotation.textContent + '$$' : null;
  };

  // Tag each live node with a stable key, then read that key back off the
  // clone. This maps every (possibly partial) cloned math to its full live
  // source by identity — not by fragile document-order index alignment, which
  // breaks the moment the cloned and intersecting node lists diverge. The full
  // LaTeX is pulled from the live node so a partially-selected expression still
  // yields its complete source.
  liveMath.forEach((m, i) => m.setAttribute('data-tex-key', i));
  try {
    // Clone the selection to preserve the surrounding prose, then swap each
    // math node for the full LaTeX of its live counterpart.
    const fragment = range.cloneContents();
    const clonedMath = fragment.querySelectorAll('math');

    if (clonedMath.length > 0) {
      clonedMath.forEach(math => {
        const key = math.getAttribute('data-tex-key');
        const source = (key !== null && liveMath[Number(key)]) || math;
        const tex = texOf(source);
        if (tex) math.replaceWith(tex);
      });
      e.preventDefault();
      e.clipboardData.setData('text/plain', fragment.textContent);
    } else {
      // The selection sits entirely inside the math, so cloneContents() copied
      // only the math's inner nodes — the <math> element itself (the common
      // ancestor) never made it into the fragment. There's no surrounding prose
      // to preserve here, so just emit the full source of each touched
      // expression.
      const tex = touched.map(texOf).filter(Boolean).join(' ');
      if (tex) {
        e.preventDefault();
        e.clipboardData.setData('text/plain', tex);
      }
    }
  } finally {
    // Don't leave the bookkeeping attribute on the live DOM.
    liveMath.forEach(m => m.removeAttribute('data-tex-key'));
  }
});

// Double-clicking inside a rendered equation selects the whole <math> block,
// mirroring how double-clicking prose selects a word.
document.addEventListener('dblclick', function(e) {
  const math = e.target.closest && e.target.closest('math');
  if (!math) return;

  const selection = window.getSelection();
  if (!selection) return;

  const range = document.createRange();
  range.selectNode(math);
  selection.removeAllRanges();
  selection.addRange(range);
});

// A long *inline* equation can be wider than its container on a narrow screen.
// CSS can't make just the overflowing ones scroll (any `overflow` rule shifts
// the baseline of every inline equation), so measure each one and tag only
// those that actually overflow; main.scss then makes those a scrollable
// inline-block via `.katex--scroll`.
(function () {
  // Content width of the nearest block-level ancestor — the width an inline
  // equation has to fit within before it overflows.
  function availableWidth(el) {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const cs = getComputedStyle(p);
      if (!cs.display.includes('inline')) {
        return p.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      }
    }
    return Infinity;
  }

  function markOverflowingInlineMath() {
    document.querySelectorAll('.katex').forEach(katex => {
      const math = katex.querySelector('math');
      if (!math || math.getAttribute('display') === 'block') return;

      // Clear any prior mark so the equation reverts to its natural width — a
      // resize that now fits should drop the scroll box again.
      katex.classList.remove('katex--scroll');
      if (katex.getBoundingClientRect().width > availableWidth(katex) + 1) {
        katex.classList.add('katex--scroll');
      }
    });
  }

  markOverflowingInlineMath();
  // Re-check once everything (incl. fonts) has settled, and on resize.
  window.addEventListener('load', markOverflowingInlineMath);
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(markOverflowingInlineMath, 150);
  });
})();
