// Site-wide language menu (EN / ไทย / 日本語), loaded with `defer` on every page.
// - puts a language button in the site nav, next to the theme toggle;
// - remembers the reader's choice (localStorage "hxh-lang") and, when a default (English)
//   address is opened later, shows that page in the chosen language if it exists;
// - points the site's own links (Reader, How it works, Deep dive, DSi) at the pages in the language
//   of the current page.
// Which pages exist in which language is listed in AVAIL; keep it in step with the files.
(() => {
  const AVAIL = {
    reader: ["en", "th", "ja"],
    explainer: ["en", "th", "ja"],
    deep: ["en", "th", "ja"],
    dsi: ["en", "th", "ja"],
  };
  const LANGS = [["en", "English", "EN"], ["th", "ไทย", "ไทย"], ["ja", "日本語", "日本語"]];
  const LABEL = { en: "Language", th: "ภาษา", ja: "言語" };
  const KEY = "hxh-lang";
  const get = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const set = (v) => { try { localStorage.setItem(KEY, v); } catch { /* private mode */ } };

  // which page family a path is, and the site root it lives under
  const parse = (path) => {
    const m = path.match(/^(.*?\/)(blog\/)?(?:(index|deep-dive|dsi)(?:-(th|ja))?(?:\.html)?)?$/);
    if (!m) return null;
    const [, root, blog, base, lang] = m;
    if (!blog && (base === "deep-dive" || base === "dsi")) return null;
    const family = blog ? (base === "deep-dive" ? "deep" : base === "dsi" ? "dsi" : "explainer") : "reader";
    return { root, family, lang: lang || "en" };
  };
  // clean URLs on the live site (it drops ".html"), file names everywhere else
  const clean = !/\.html$/.test(location.pathname) && !/^(127\.0\.0\.1|localhost|\[::1\])$/.test(location.hostname);
  const urlOf = (root, family, lang) => {
    const sfx = lang === "en" ? "" : `-${lang}`;
    const dir = family === "reader" ? root : `${root}blog/`;
    const base = family === "deep" ? "deep-dive" : family === "dsi" ? "dsi" : "index";
    if (clean) return base === "index" && !sfx ? dir : `${dir}${base}${sfx}`;
    return `${dir}${base}${sfx}.html`;
  };

  const here = parse(location.pathname);
  if (!here) return;
  const pageLang = (document.documentElement.lang || "en").slice(0, 2);
  const avail = (f, l) => (AVAIL[f] || []).includes(l);

  // a remembered choice applies to the default (English) addresses, if the page exists in that
  // language; an address that names a language (…-th, …-ja) is always shown as asked
  const pref = get();
  if (pref && pageLang === "en" && pref !== "en" && avail(here.family, pref)) {
    location.replace(urlOf(here.root, here.family, pref) + location.hash);
    return;
  }

  const ready = (fn) => (document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", fn) : fn());
  ready(() => {
    // the site's own links follow the page language
    for (const a of document.querySelectorAll("a[href]")) {
      if (a.closest(".lang-menu")) continue;
      const u = new URL(a.getAttribute("href"), location.href);
      if (u.origin !== location.origin) continue;
      const t = parse(u.pathname);
      if (!t) continue;
      const lang = avail(t.family, pageLang) ? pageLang : "en";
      if (lang !== t.lang) a.setAttribute("href", urlOf(t.root, t.family, lang) + u.hash);
    }

    // the menu
    const nav = document.querySelector(".site-nav .nav-inner");
    if (!nav) return;
    const box = document.createElement("div");
    box.className = "lang-menu";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lang-btn";
    btn.setAttribute("aria-haspopup", "true");
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-label", `${LABEL[pageLang] || LABEL.en}: ${(LANGS.find((l) => l[0] === pageLang) || LANGS[0])[1]}`);
    btn.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.6 3.9 5.6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-5.6-3.9-9S9.4 5.6 12 3z"/></svg><span></span>`;
    btn.querySelector("span").textContent = (LANGS.find((l) => l[0] === pageLang) || LANGS[0])[2];
    const list = document.createElement("ul");
    list.className = "lang-list";
    list.hidden = true;
    for (const [code, name] of LANGS) {
      const li = document.createElement("li");
      const ok = avail(here.family, code);
      const a = document.createElement(ok ? "a" : "span");
      a.textContent = name;
      a.lang = code;
      if (ok) {
        a.href = urlOf(here.root, here.family, code) + location.hash;
        a.hreflang = code;
        if (code === pageLang) a.setAttribute("aria-current", "page");
        a.addEventListener("click", () => set(code));
      } else {
        a.className = "soon";
        a.title = "Translation coming soon";
      }
      li.append(a);
      list.append(li);
    }
    box.append(btn, list);
    const toggle = document.getElementById("theme-toggle");
    if (toggle && toggle.parentNode === nav) nav.insertBefore(box, toggle); else nav.append(box);

    const open = (v) => {
      list.hidden = !v;
      btn.setAttribute("aria-expanded", String(v));
      if (v) (list.querySelector("a[aria-current]") || list.querySelector("a"))?.focus();
    };
    btn.addEventListener("click", () => open(list.hidden));
    document.addEventListener("click", (e) => { if (!box.contains(e.target)) open(false); });
    box.addEventListener("keydown", (e) => {
      const items = [...list.querySelectorAll("a")];
      const i = items.indexOf(document.activeElement);
      if (e.key === "Escape") { open(false); btn.focus(); }
      else if (e.key === "ArrowDown" && !list.hidden) { e.preventDefault(); items[(i + 1) % items.length]?.focus(); }
      else if (e.key === "ArrowUp" && !list.hidden) { e.preventDefault(); items[(i - 1 + items.length) % items.length]?.focus(); }
    });
  });
})();
