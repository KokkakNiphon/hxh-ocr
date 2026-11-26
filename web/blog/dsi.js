// The DSi post (dsi.html and its translations): three bar charts and the figure/table numbers.
// Numbers are copied from dsi/docs/research/JOURNAL.md ("the DSi reader on the DS, beside the Fast
// reader") and docs/research/JOURNAL.md ("dsi_v1, a 3-stage 32 px reader for the DSi").
// The chart text is here in each page language, so the translated pages need no i18n file.

const L = {
  en: {
    readT: "Seconds to read one line in melonDS",
    readS: "original-DS mode, 67 MHz, the same test line (lower is better)",
    plain: "plain code", kernel: "ARM9 kernel",
    fast: "Fast reader (web)\n154M multiply-adds", dsi: "DSi reader, dsi_v1\n30.6M multiply-adds",
    cerT: "Character error rate, int8",
    cerS: "share of characters read wrong, on generated lines (lower is better)",
    fastS: "Fast reader (web)", dsiS: "DSi reader, dsi_v1",
    digits: "lines with numbers\nheld-out test", v6: "lines with numbers\nv6 validation",
    photo: "photo-style lines", clean: "clean lines",
    opsT: "Cycles per multiply-add on the ARM9",
    opsS: "the DSi reader in melonDS, by kind of layer: convolutions with the ARM9 kernels, depthwise in plain code and with its new kernel (* an estimate). Lower is better.",
    o1: "1×1 conv", ok: "larger convs", odw: "depthwise, plain", odk: "depthwise, new*",
    s: " s", c: "",
  },
  th: {
    readT: "เวลาอ่านหนึ่งบรรทัดใน melonDS (วินาที)",
    readS: "โหมด DS รุ่นแรก 67 MHz บรรทัดทดสอบเดียวกัน (ยิ่งน้อยยิ่งดี)",
    plain: "โค้ดธรรมดา", kernel: "เคอร์เนล ARM9",
    fast: "ตัวอ่านชุดเร็ว (เว็บ)\n154.2 ล้านการคูณบวก", dsi: "ตัวอ่าน DSi, dsi_v1\n30.6 ล้านการคูณบวก",
    cerT: "อัตราการอ่านตัวอักษรผิด, int8",
    cerS: "สัดส่วนตัวอักษรที่อ่านผิด บนบรรทัดที่สร้างขึ้น (ยิ่งน้อยยิ่งดี)",
    fastS: "ตัวอ่านชุดเร็ว (เว็บ)", dsiS: "ตัวอ่าน DSi, dsi_v1",
    digits: "บรรทัดที่มีตัวเลข\nชุดทดสอบที่แยกไว้", v6: "บรรทัดที่มีตัวเลข\nชุดตรวจ v6",
    photo: "บรรทัดแบบภาพถ่าย", clean: "บรรทัดสะอาด",
    opsT: "จำนวนรอบสัญญาณนาฬิกาต่อการคูณบวกบน ARM9",
    opsS: "ตัวอ่าน DSi ใน melonDS แยกตามชนิดของเลเยอร์: คอนโวลูชันใช้เคอร์เนล ARM9 ส่วน depthwise วัดทั้งแบบโค้ดธรรมดาและแบบเคอร์เนลใหม่ (* คือค่าประมาณ) ยิ่งน้อยยิ่งดี",
    o1: "conv 1×1", ok: "conv ใหญ่กว่า", odw: "depthwise ธรรมดา", odk: "depthwise ใหม่*",
    s: " วิ", c: "",
  },
  ja: {
    readT: "melonDS で 1 行を読む時間（秒）",
    readS: "初代 DS モード・67 MHz・同じテスト行（短いほど良い）",
    plain: "通常のコード", kernel: "ARM9 カーネル",
    fast: "「高速」（Web）\n積和 1億5420万回", dsi: "DSi リーダー dsi_v1\n積和 3060万回",
    cerT: "文字誤り率（int8）",
    cerS: "生成した行で読み誤った文字の割合（低いほど良い）",
    fastS: "「高速」リーダー（Web）", dsiS: "DSi リーダー dsi_v1",
    digits: "数字を含む行\n未使用のテスト", v6: "数字を含む行\nv6 検証",
    photo: "写真風の行", clean: "きれいな行",
    opsT: "ARM9 での積和 1 回あたりのサイクル数",
    opsS: "melonDS 上の DSi リーダー、層の種類別。畳み込みは ARM9 カーネル、depthwise は通常のコードと新カーネル（* は推定値）。少ないほど良い",
    o1: "1×1 畳み込み", ok: "大きい畳み込み", odw: "depthwise・通常", odk: "depthwise・新*",
    s: " 秒", c: "",
  },
};

const lang = (document.documentElement.lang || "en").slice(0, 2);
const T = L[lang] || L.en;

function charts() {
  const { barChart } = window.HXH;
  const $ = (id) => document.getElementById(id);
  const accent = "var(--accent)", alt = "var(--line-strong)";

  barChart($("chart-read"), {
    title: T.readT, sub: T.readS, unit: T.s, digits: 2,
    series: [{ name: T.plain, color: alt }, { name: T.kernel, color: accent }],
    rows: [
      { label: T.fast, values: [29.57, 11.01] },
      { label: T.dsi, values: [6.06, 2.38] },
    ],
  });

  barChart($("chart-cer"), {
    title: T.cerT, sub: T.cerS, unit: "%", digits: 2,
    series: [{ name: T.fastS, color: alt }, { name: T.dsiS, color: accent }],
    rows: [
      { label: T.digits, values: [24.84, 13.35] },
      { label: T.v6, values: [20.10, 12.01] },
      { label: T.photo, values: [9.22, 9.89] },
      { label: T.clean, values: [1.50, 2.43] },
    ],
  });

  barChart($("chart-ops"), {
    title: T.opsT, sub: T.opsS, digits: 1,
    rows: [
      { label: T.o1, value: 3.7 },
      { label: T.ok, value: 4.2 },
      { label: T.odw, value: 18.8 },
      { label: T.odk, value: 3.5, hi: true },
    ],
  });
}

/** Number figures and tables in document order ("Fig. # —", "Table # —" in each page's own words). */
function numberAll() {
  const FIG = document.documentElement.dataset.fig || "Fig.", TAB = document.documentElement.dataset.table || "Table";
  let f = 0, t = 0;
  for (const fig of document.querySelectorAll("article figure")) {
    const fc = fig.querySelector("figcaption");
    if (!fc || fig.hidden) continue;
    fc.dataset.raw ??= fc.textContent;
    const raw = fc.dataset.raw;
    if (raw.startsWith(`${FIG} # —`)) fc.textContent = raw.replace("#", String(++f));
    else if (raw.startsWith(`${TAB} # —`)) fc.textContent = raw.replace("#", String(++t));
  }
}

const go = () => {
  try { charts(); } catch (e) { console.warn("dsi.js: charts failed", e); }
  numberAll();
};
if (window.HXH) go();
else document.addEventListener("hxh:ready", go, { once: true });
