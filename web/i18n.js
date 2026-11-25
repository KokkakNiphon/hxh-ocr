// Runtime translator for the reader page in Thai and Japanese (index-th.html, index-ja.html).
// The page loads its dictionary first (i18n-reader-xx.js sets window.HXH_I18N = { strings, patterns }),
// then this script. Every text node and label attribute under <body> whose trimmed text matches a
// string exactly, or else the first matching pattern, is replaced, now and whenever app.js changes
// the page. Anything inside [data-no-i18n] (readings, file names) is left alone.
// Same rules as the blog's translator (web/blog/blog.js startTranslator).
(() => {
  const start = () => {
    const T = window.HXH_I18N;
    if (!T) return;
    const strings = T.strings || {};
    const pats = (T.patterns || []).map(([re, out]) => [new RegExp(re), out]);
    const made = new Set();
    const tr = (s) => {
      const t = s.trim();
      if (!t || made.has(t)) return null;
      let out = strings[t];
      if (out == null) for (const [re, rep] of pats) if (re.test(t)) { out = t.replace(re, rep); break; }
      if (out == null || out === t) return null;
      made.add(out);
      return s.replace(t, out);
    };
    const ATTRS = ["aria-label", "alt", "title", "placeholder"];
    const skip = (n) => n.closest?.("[data-no-i18n], script, style");
    const walk = (node) => {
      if (node.nodeType === 3) {
        if (node.parentElement && skip(node.parentElement)) return;
        const v = tr(node.data);
        if (v != null) node.data = v;
        return;
      }
      if (node.nodeType !== 1 || skip(node)) return;
      for (const a of ATTRS) {
        const v = node.getAttribute?.(a);
        if (v) { const o = tr(v); if (o != null) node.setAttribute(a, o); }
      }
      for (const c of node.childNodes) walk(c);
    };
    walk(document.body);
    new MutationObserver((muts) => {
      for (const m of muts) {
        if (m.type === "characterData" || m.type === "attributes") walk(m.target);
        else m.addedNodes.forEach(walk);
      }
    }).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
