/**
 * index.htm – client-side interactions
 * - GOV.UK radio styling sync
 * - Clause drawer toggles
 * - File upload UI helpers
 * - Form validation (at least one radio)
 * - Fancybox binding
 * - Interstitial loading screen + cycling messages + timed redirect (only if valid)
 */

document.addEventListener("DOMContentLoaded", function () {

  // ================================================================
  // =============== GOV.UK RADIOS: CHECKED STATE SYNC ===============
  // Adds/removes a CSS class on the radio's parent to reflect checked state
  // ================================================================
  const radios = document.querySelectorAll(".govuk-radios__input");

  radios.forEach(radio => {
    // Run once on page load (to set initial state)
    toggleCheckedClass(radio);

    // Re-run whenever a radio in the same group changes
    radio.addEventListener("change", () => {
      const groupName = radio.name;
      const group = document.querySelectorAll(`.govuk-radios__input[name="${groupName}"]`);
      group.forEach(r => toggleCheckedClass(r));
    });
  });

  // Adds/removes a CSS class to highlight the checked radio's parent
  function toggleCheckedClass(radio) {
    const item = radio.closest(".govuk-radios__item");
    if (!item) return;

    if (radio.checked) {
      item.classList.add("govuk-radios__item--checked");
    } else {
      item.classList.remove("govuk-radios__item--checked");
    }
  }


  // ================================================================
  // ====================== CLAUSE DRAWER TOGGLE =====================
  // Toggles an "open" class on the parent .clause and updates the H4 text
  // ================================================================
  document.querySelectorAll(".clause-toggle .emissions-toggle").forEach(toggle => {
    toggle.addEventListener("click", () => {
      const toggleWrap = toggle.closest(".clause-toggle");
      const article = toggle.closest(".clause");
      const inner = toggleWrap.querySelector(".clause-toggle-inner");

      const isOpen = article.classList.toggle("open");

      // Update heading text
      toggle.textContent = isOpen
        ? "Hide emissions"
        : "Show emissions";
    });
  });


  // ================================================================
  // ======================= FILE UPLOAD HANDLING ====================
  // Shows selected filenames, updates label text, and (optionally) selects a radio
  // ================================================================
  const fileInput  = document.getElementById('file');
  const filenameEl = document.querySelector('.upload__filename');
  const labelEl    = document.querySelector('label[for="file"]'); // "Choose a file" button
  const radioEl    = document.getElementById('Contract5');        // optional

  // If there’s no file input on this page, just skip this block (don’t return)
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const hasFiles = fileInput.files && fileInput.files.length > 0;

      // Show/hide filename (if the element is present)
      if (filenameEl) {
        if (hasFiles) {
          const names = Array.from(fileInput.files).map(f => f.name).join(', ');
          filenameEl.innerHTML = `<strong>File to analyse:</strong><br /> ${names}`;
          filenameEl.style.display = 'block';
        } else {
          filenameEl.textContent = '';
          filenameEl.style.display = 'none';
        }
      }

      // Update label text (if present)
      if (labelEl) {
        labelEl.textContent = hasFiles ? 'Choose another file' : 'Choose file';
      }

      // Select radio (if present)
      if (hasFiles && radioEl) {
        radioEl.checked = true;
        radioEl.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  }


  // ================================================================
  // ========================= FORM VALIDATION =======================
  // Ensures at least one Contract radio is selected; accessible error UX
  // Exposed helpers so other handlers (like the loading button) can reuse.
  // ================================================================
  const form   = document.getElementById('analysisForm');
  const group  = document.getElementById('analysis-group');
  const error  = document.getElementById('analysis-error');
  const contractRadios = Array.from(document.querySelectorAll('input[name="Contract"]'));

  // Check if any radio is selected
  function anyChecked() {
    return contractRadios.some(r => r.checked);
  }

  // Show validation error styling and message
  function showError() {
    if (!error || !group || contractRadios.length === 0) return;

    error.hidden = false;
    group.classList.add('govuk-form-group--error');

    // Mark radios as invalid (associate first with the error for AT)
    contractRadios.forEach(r => r.setAttribute('aria-invalid', 'true'));
    contractRadios[0].setAttribute(
      'aria-describedby',
      [contractRadios[0].getAttribute('aria-describedby'), 'analysis-error'].filter(Boolean).join(' ')
    );

    // Move focus to first radio for quick correction
    contractRadios[0].focus();
  }

  // Clear validation error styling and message
  function clearError() {
    if (!error || !group || contractRadios.length === 0) return;

    error.hidden = true;
    group.classList.remove('govuk-form-group--error');
    contractRadios.forEach(r => {
      r.removeAttribute('aria-invalid');
      // Remove the error id from aria-describedby but keep any existing hint id
      const desc = (r.getAttribute('aria-describedby') || '')
        .split(' ')
        .filter(id => id && id !== 'analysis-error')
        .join(' ');
      if (desc) r.setAttribute('aria-describedby', desc);
      else r.removeAttribute('aria-describedby');
    });
  }

  // Validate on submit – if valid, we’ll let the loading handler take over.
  if (form) {
    form.addEventListener('submit', function (e) {
      if (!anyChecked()) {
        e.preventDefault();
        showError();
      } else {
        // If you want real submission AND loading screen, do not preventDefault here.
        // Leave default behavior and trigger loading in a 'submit' listener (below) or via the button.
      }
    });
  }

  // Clear error as soon as a selection is made/changed
  contractRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      if (anyChecked()) clearError();
    });
  });


  // ================================================================
  // ========================= FANCYBOX BINDING ======================
  // Initialises Fancybox for any element with data-fancybox
  // ================================================================
  Fancybox.bind('[data-fancybox]', {
    groupAll: false // don't group all items into one gallery
  });


  // ================================================================
  // ============ INTERSTITIAL LOADING UI (ONLY IF VALID) ===========
  // Shows a loading screen, cycles status messages, then redirects
  // ================================================================
  const loadingScreen = document.getElementById("loading-screen");
  const button        = document.getElementById("getInsights");

  if (button) {
    button.addEventListener("click", (e) => {
      e.preventDefault(); // stop instant form submission

      if (!anyChecked()) {
        // Block loading and show validation error
        showError();
        return;
      }

      // Passed validation — proceed with interstitial loading
      if (loadingScreen) {
        loadingScreen.classList.add("active");
      }

      // Start cycling through loading messages
      startLoadingStates();

      // Redirect after 9 seconds based on which radio was chosen
      setTimeout(() => {
        const chosen = contractRadios.find(r => r.checked);
        if (chosen) {
          // e.g. "Contract1" → "contract-1.htm"
          const page = chosen.id.toLowerCase().replace("contract", "contract-report-") + ".htm";
          window.location.href = page;
        } else {
          // fallback if somehow no radio is checked
          window.location.href = "contract-report-1.htm";
        }
      }, 12000);
    });
  }

  // Animate through loading messages (cycling states with fade in/out)
  function startLoadingStates() {
    const states = document.querySelectorAll(".loading-content-wrap");
    if (!states.length) return;

    let current = 0;

    // Reset states
    states.forEach(s => s.classList.remove("active", "fade-out"));
    states[current].classList.add("active");

    setInterval(() => {
      states[current].classList.remove("active");
      states[current].classList.add("fade-out");

      const next = (current + 1) % states.length;

      setTimeout(() => {
        states[current].classList.remove("fade-out");
        states[next].classList.add("active");
        current = next;
      }, 600); // match CSS transition
    }, 4000);
  }

  // Add a slight delay so transition kicks in after initial paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.gauge').forEach(el => {
      el.classList.add('loaded');
    });
  });


  // ================================================================
  // ====================== SMOOTH SCROLL TO ANCHOR ==================
  // Smoothly scrolls the page to any anchor link target
  // ================================================================
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function (e) {
      const targetId = this.getAttribute("href");
      if (targetId.length > 1) { // ignore plain "#"
        const target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: "smooth",
            block: "start"
          });
        }
      }
    });
  });

});
