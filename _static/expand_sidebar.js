document.addEventListener('DOMContentLoaded', function () {
  try {
    // Find all expand toggles used by the RTD theme and open them if collapsed
    var toggles = document.querySelectorAll('.toctree-expand');
    toggles.forEach(function (btn) {
      try {
        var expanded = btn.getAttribute('aria-expanded');
        if (expanded === 'false') btn.click();
      } catch (e) { /* ignore */ }
    });
  } catch (e) {
    // fail silently
    console.debug('expand_sidebar.js error', e);
  }
});
