// Basic script
document.addEventListener("DOMContentLoaded", function () {
  const radios = document.querySelectorAll(".govuk-radios__input");

  radios.forEach(radio => {
    // Run once on page load (to set initial state)
    toggleCheckedClass(radio);

    // Listen for change events
    radio.addEventListener("change", () => {
      // For radios, only one in the group can be checked
      const groupName = radio.name;
      const group = document.querySelectorAll(`.govuk-radios__input[name="${groupName}"]`);

      group.forEach(r => toggleCheckedClass(r));
    });
  });

  function toggleCheckedClass(radio) {
    const item = radio.closest(".govuk-radios__item");
    if (!item) return;

    if (radio.checked) {
      item.classList.add("govuk-radios__item--checked");
    } else {
      item.classList.remove("govuk-radios__item--checked");
    }
  }

  document.querySelectorAll(".clause-toggle h4").forEach(toggle => {
    toggle.addEventListener("click", () => {
      const toggleWrap = toggle.closest(".clause-toggle");
      const article = toggle.closest(".clause");
      const inner = toggleWrap.querySelector(".clause-toggle-inner");

      const isOpen = article.classList.toggle("open");

      // Update heading text
      if (isOpen) {
        toggle.textContent = "Hide emissions and how they're addressed";
      } else {
        toggle.textContent = "Show emissions and how they're addressed";
      }
    });
  });


  const fileInput = document.getElementById('file');
  const filenameEl = document.querySelector('.upload__filename');
  const labelEl = document.querySelector('label[for="file"]'); // "Choose a file" button
  const radioEl = document.getElementById('Contract5');        // optional

  // If there’s no file input on this page, do nothing.
  if (!fileInput) return;

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



  (function () {
    const form = document.getElementById('analysisForm');
    const group = document.getElementById('analysis-group');
    const error = document.getElementById('analysis-error');
    const radios = Array.from(document.querySelectorAll('input[name="Contract"]'));

    function anyChecked() {
      return radios.some(r => r.checked);
    }

    function showError() {
      error.hidden = false;
      group.classList.add('govuk-form-group--error');

      // Mark radios as invalid (associate first with the error for AT)
      radios.forEach(r => r.setAttribute('aria-invalid', 'true'));
      radios[0].setAttribute('aria-describedby',
        [radios[0].getAttribute('aria-describedby'), 'analysis-error'].filter(Boolean).join(' ')
      );

      // Move focus to first radio for quick correction
      radios[0].focus();
    }

    function clearError() {
      error.hidden = true;
      group.classList.remove('govuk-form-group--error');
      radios.forEach(r => {
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

    // Validate on submit
    form.addEventListener('submit', function (e) {
      if (!anyChecked()) {
        e.preventDefault();
        showError();
      }
    });

    // Clear error as soon as a selection is made/changed
    radios.forEach(radio => {
      radio.addEventListener('change', () => {
        if (anyChecked()) clearError();
      });
    });
  })();

  // Bind Fancybox to any element with the data-fancybox attribute
  Fancybox.bind('[data-fancybox]', {
    // optional options
    groupAll: false
  });


});
