
(function () {
  const POPUP_ID = 'footnote-popup';

  // Visual styling lives in assets/main.scss (#footnote-popup) so it tracks the
  // theme's colours. JS owns only the dynamic bits: toggling `display` to
  // show/hide, and setting `left`/`top` while positioning.
  const popup = document.createElement('div');
  popup.id = POPUP_ID;
  popup.setAttribute('role', 'tooltip');
  document.body.appendChild(popup);

  function getFootnoteContent(href) {
    const id = href.replace(/^#/, '');
    const li = document.getElementById(id);
    if (!li) return null;

    const clone = li.cloneNode(true);
    clone.querySelectorAll('.reversefootnote').forEach(el => el.remove());
    return clone.innerHTML.trim();
  }

  function positionPopup(anchor) {
    const rect = anchor.getBoundingClientRect();
    const margin = 8;

    let top = rect.bottom + margin;
    let left = rect.left;

    const popupWidth = popup.offsetWidth;
    const popupHeight = popup.offsetHeight;

    if (left + popupWidth > window.innerWidth - margin) {
      left = window.innerWidth - margin - popupWidth;
    }
    if (left < margin) {
      left = margin;
    }
    if (top + popupHeight > window.innerHeight - margin) {
      top = rect.top - margin - popupHeight;
    }

    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
  }

  // Brief grace period so the cursor can travel from the anchor across the
  // gap to the popup without the popup vanishing in between.
  const HIDE_DELAY = 200;
  let hideTimer = null;

  let activeAnchor = null;
  // Set while we move focus back to an anchor programmatically, so the
  // anchor's own focus handler doesn't immediately re-open the popup.
  let suppressShow = false;

  function getFocusable(container) {
    const selector = 'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])';
    return Array.from(container.querySelectorAll(selector))
      .filter(el => !el.disabled && el.offsetParent !== null);
  }

  function refocusAnchor(anchor) {
    if (!anchor) return;
    suppressShow = true;
    anchor.focus();
    suppressShow = false;
  }

  function cancelHide() {
    if (hideTimer !== null) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  function hidePopup() {
    popup.style.display = 'none';
    if (activeAnchor) {
      activeAnchor.removeAttribute('aria-describedby');
      activeAnchor = null;
    }
  }

  function scheduleHide() {
    cancelHide();
    hideTimer = setTimeout(hidePopup, HIDE_DELAY);
  }

  function showPopup(anchor) {
    const href = anchor.getAttribute('href');
    if (!href) return;

    const content = getFootnoteContent(href);
    if (!content) return;

    cancelHide();
    if (activeAnchor && activeAnchor !== anchor) {
      activeAnchor.removeAttribute('aria-describedby');
    }

    popup.innerHTML = content;
    popup.style.display = 'block';

    // The popup (≤320px) is often narrower than the page content area, so an
    // equation that fit on the page — and was never tagged .katex--scroll —
    // may still overflow here. Re-run the detection scoped to the popup using
    // its actual content width. Do this before positionPopup so that any
    // added padding-bottom (from .katex--scroll) is included in offsetHeight.
    const popupCs = getComputedStyle(popup);
    const popupContentWidth = popup.clientWidth
      - parseFloat(popupCs.paddingLeft) - parseFloat(popupCs.paddingRight);
    popup.querySelectorAll('.katex').forEach(function (katex) {
      const math = katex.querySelector('math');
      if (!math || math.getAttribute('display') === 'block') return;
      katex.classList.remove('katex--scroll');
      if (katex.getBoundingClientRect().width > popupContentWidth + 1) {
        katex.classList.add('katex--scroll');
      }
    });

    positionPopup(anchor);

    activeAnchor = anchor;
    // Screen readers announce the preview as the link's description.
    anchor.setAttribute('aria-describedby', POPUP_ID);
  }

  // Track the input type, and whether the popup was already open, as of the
  // start of an interaction — so a tap can toggle the preview in place rather
  // than navigating to the note the way a mouse click does.
  let lastPointerType = 'mouse';
  let openAnchorAtPointerdown = null;
  document.addEventListener('pointerdown', function (e) {
    lastPointerType = e.pointerType || 'mouse';
    openAnchorAtPointerdown = activeAnchor;
  }, true);

  // Keep the popup open while the cursor is over it; hide once it leaves.
  popup.addEventListener('mouseenter', cancelHide);
  popup.addEventListener('mouseleave', scheduleHide);

  // Keep it open while keyboard focus is inside it; hide when focus leaves.
  popup.addEventListener('focusin', cancelHide);
  popup.addEventListener('focusout', function (e) {
    if (!popup.contains(e.relatedTarget)) scheduleHide();
  });

  // Trap Tab within the popup: forward Tab past the last element wraps to the
  // first, and Shift+Tab past the first returns to the originating link.
  popup.addEventListener('keydown', function (e) {
    if (e.key !== 'Tab') return;

    const focusable = getFocusable(popup);
    if (!focusable.length) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      refocusAnchor(activeAnchor);
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  });

  document.querySelectorAll('a.footnote').forEach(function (anchor) {
    if (!anchor.getAttribute('href')) return;

    // Hover (mouse) and keyboard focus both reveal the popup.
    anchor.addEventListener('mouseenter', function () { showPopup(anchor); });
    anchor.addEventListener('mouseleave', scheduleHide);
    anchor.addEventListener('focus', function () {
      if (!suppressShow) showPopup(anchor);
    });
    anchor.addEventListener('blur', scheduleHide);

    // Tab from the anchor moves focus into the popup's content (if any),
    // letting keyboard users reach links inside a footnote.
    anchor.addEventListener('keydown', function (e) {
      if (e.key !== 'Tab' || e.shiftKey) return;
      if (activeAnchor !== anchor) return;

      const focusable = getFocusable(popup);
      if (!focusable.length) return;

      e.preventDefault();
      cancelHide();
      focusable[0].focus();
    });

    // Touch/pen tap toggles the preview instead of jumping to the note.
    anchor.addEventListener('click', function (e) {
      if (lastPointerType === 'mouse') return;
      e.preventDefault();
      if (openAnchorAtPointerdown === anchor) {
        hidePopup();
      } else {
        showPopup(anchor);
      }
    });
  });

  // Escape dismisses the popup and returns focus to the footnote link.
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && activeAnchor) {
      const anchor = activeAnchor;
      cancelHide();
      hidePopup();
      refocusAnchor(anchor);
    }
  });

  // A tap/click anywhere outside the popup and the footnote links dismisses it.
  document.addEventListener('click', function (e) {
    if (!activeAnchor) return;
    if (popup.contains(e.target)) return;
    if (e.target.closest && e.target.closest('a.footnote')) return;
    hidePopup();
  });
})();
