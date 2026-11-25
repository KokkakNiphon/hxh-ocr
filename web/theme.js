// Theme for every page: auto (follows the system), light or dark, remembered in this browser.
// Loaded as a classic script in <head> so the page paints in the chosen theme straight away;
// the nav button (#theme-toggle) cycles auto -> light -> dark and fires a "hxh:theme" event
// for scripts that draw with theme colours (the reader's canvas).
(function () {
  var KEY = "hxh-theme";
  var ORDER = ["", "light", "dark"];
  var root = document.documentElement;
  function get() { try { return localStorage.getItem(KEY) || ""; } catch (e) { return ""; } }
  function put(v) { try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); } catch (e) { /* private mode */ } }
  function apply(v) { if (v) root.setAttribute("data-theme", v); else root.removeAttribute("data-theme"); }
  apply(get());

  function wire() {
    var b = document.getElementById("theme-toggle");
    if (!b) return;
    var names = { "": b.getAttribute("data-auto") || "auto", light: b.getAttribute("data-light") || "light", dark: b.getAttribute("data-dark") || "dark" };
    var prefix = b.getAttribute("data-prefix") || "Theme";
    function render() {
      var v = get();
      b.setAttribute("data-mode", v || "auto");
      b.setAttribute("aria-label", prefix + ": " + names[v]);
      b.title = prefix + ": " + names[v];
    }
    b.addEventListener("click", function () {
      var v = ORDER[(ORDER.indexOf(get()) + 1) % ORDER.length];
      put(v);
      apply(v);
      render();
      window.dispatchEvent(new Event("hxh:theme"));
    });
    render();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
  else wire();
})();
