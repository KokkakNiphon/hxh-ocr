// Thai translations of the text web/app.js shows on the reader page (index-th.html). Loaded as a
// classic script before i18n.js, which replaces any text node or aria-label/alt/title/placeholder
// whose trimmed text matches a string here exactly, else the first matching pattern.
// Readings (kana, romaji) and numbers are never translated: every pattern is anchored on English
// words that a reading cannot contain.
(function () {
  var strings = {
    // ------------------------------------------------------------ app.js MSG (toasts)
    "This browser could not open that image. Try a JPEG or PNG.": "เบราว์เซอร์นี้เปิดรูปนั้นไม่ได้ ลองใช้ไฟล์ JPEG หรือ PNG",
    "That file isn't an image. Choose, paste or drop a JPEG, PNG or WebP picture.": "ไฟล์นั้นไม่ใช่รูปภาพ ให้เลือก วาง หรือลากรูป JPEG, PNG หรือ WebP มาวาง",
    "There's no image on the clipboard. Copy an image first, then try again.": "ไม่มีรูปภาพในคลิปบอร์ด คัดลอกรูปก่อน แล้วลองอีกครั้ง",
    "The browser didn't let this page read the clipboard. Paste with the keyboard (Ctrl+V or ⌘V), or choose the image instead.": "เบราว์เซอร์ไม่อนุญาตให้หน้านี้อ่านคลิปบอร์ด ให้วางด้วยคีย์บอร์ด (Ctrl+V หรือ ⌘V) หรือเลือกรูปแทน",
    "That link isn't an image. Drag the picture itself, or save it and choose the file.": "ลิงก์นั้นไม่ใช่รูปภาพ ให้ลากตัวรูปภาพเอง หรือบันทึกรูปแล้วเลือกไฟล์",
    "The site that image comes from doesn't let other pages read it. Save the image to your device, then drop or choose the file.": "เว็บไซต์ต้นทางของรูปนั้นไม่อนุญาตให้หน้าอื่นอ่านรูป บันทึกรูปลงอุปกรณ์ก่อน แล้วลากมาวางหรือเลือกไฟล์",
    "Couldn't copy. Select the text and copy it yourself.": "คัดลอกไม่สำเร็จ ลองเลือกข้อความแล้วคัดลอกเอง",

    // ------------------------------------------------------------ status line, model info, buttons
    "All int8, WebAssembly SIMD.": "ทั้งหมดเป็น int8 บน WebAssembly SIMD",
    "All int8, WebAssembly (no SIMD).": "ทั้งหมดเป็น int8 บน WebAssembly (ไม่มี SIMD)",
    "The accurate models could not be loaded, so Fast is used.": "โหลดโมเดลชุดแม่นยำไม่สำเร็จ จึงใช้ชุดเร็วแทน",
    "Ready. Everything runs on this device.": "พร้อมใช้งาน ทุกอย่างทำงานบนอุปกรณ์นี้",
    "Live mode needs the line detector, which did not load.": "โหมดสดต้องใช้ตัวตรวจจับบรรทัด ซึ่งโหลดไม่สำเร็จ",
    "Show all": "แสดงทั้งหมด",
    "Zoom to line": "ซูมไปที่บรรทัด",

    // ------------------------------------------------------------ hint under the picture, notes
    "Tap a numbered line to see it. Tap unmarked text to look closer, or drag a box around one line to read just that.": "แตะบรรทัดที่มีหมายเลขเพื่อดู แตะข้อความที่ยังไม่มีกรอบเพื่อให้มองใกล้ขึ้น หรือลากกรอบล้อมบรรทัดเดียวเพื่ออ่านเฉพาะบรรทัดนั้น",
    "Drag a box around one line of text to read it.": "ลากกรอบล้อมข้อความหนึ่งบรรทัดเพื่ออ่าน",
    "Looking for a line here…": "กำลังหาบรรทัดตรงนี้…",
    "No line found at that spot. Drag a box around the text to read it anyway.": "ไม่พบบรรทัดตรงนั้น ลากกรอบล้อมข้อความเพื่ออ่านได้เลย",

    // ------------------------------------------------------------ page summary
    "Finding the lines…": "กำลังหาบรรทัด…",
    "Drag a box around one line of text.": "ลากกรอบล้อมข้อความหนึ่งบรรทัด",

    // ------------------------------------------------------------ line list: tags and empty readings
    "(nothing recognised)": "(ไม่พบตัวอักษร)",
    "not Hunter script?": "ไม่ใช่อักษรฮันเตอร์?",
    "vertical": "แนวตั้ง",
    "tapped": "แตะหา",
    "box": "กรอบ",
    "not Hunter script? · vertical": "ไม่ใช่อักษรฮันเตอร์? · แนวตั้ง",
    "vertical · tapped": "แนวตั้ง · แตะหา",
    "vertical · box": "แนวตั้ง · กรอบ",

    // ------------------------------------------------------------ selected line
    "No text recognised in this line.": "ไม่พบตัวอักษรในบรรทัดนี้",
    ", runner-up": ", อันดับรองคือ",
    "Label:": "คำตอบที่ถูกต้อง:",
    "✓ match": "✓ ตรงกัน",
    "✗ differs": "✗ ไม่ตรงกัน",
    "The text is small in this image, so it was scaled up. Get closer (or use a sharper image) so the small dakuten marks survive.": "ตัวหนังสือในรูปนี้เล็ก จึงถูกขยายขึ้น เข้าไปใกล้กว่านี้ (หรือใช้รูปที่คมชัดกว่า) เพื่อให้เครื่องหมายดาคุเต็นเล็ก ๆ ยังมองเห็นได้",
    "No vertical reader is loaded, so this column was read by the horizontal one.": "ไม่ได้โหลดตัวอ่านแนวตั้ง คอลัมน์นี้จึงถูกอ่านด้วยตัวอ่านแนวนอน",
    "Auto-fit: read the box tightened around the text.": "ปรับกรอบอัตโนมัติ: อ่านกรอบที่บีบให้แนบกับตัวหนังสือแล้ว",
    "Found by looking closer around your tap.": "พบจากการมองใกล้ขึ้นรอบจุดที่คุณแตะ",

    // ------------------------------------------------------------ live camera
    "Point the camera at Hunter × Hunter script. Tap a line (or Freeze) to keep the picture.": "เล็งกล้องไปที่อักษรฮันเตอร์ × ฮันเตอร์ แตะที่บรรทัด (หรือกดหยุดภาพ) เพื่อเก็บภาพไว้",
    "Paused while the page is hidden.": "หยุดชั่วคราวขณะที่หน้านี้ถูกซ่อนอยู่",
    "Starting…": "กำลังเริ่ม…",

    // ------------------------------------------------------------ language menu (lang.js)
    "Translation coming soon": "คำแปลจะมาเร็ว ๆ นี้"
  };

  // why the camera could not be used: each sentence on its own and as the start of a longer one
  var WHY = [
    ["The camera isn't allowed here \\(blocked by the browser or by the page that hosts this demo\\)\\.",
      "ไม่ได้รับอนุญาตให้ใช้กล้องที่นี่ (ถูกบล็อกโดยเบราว์เซอร์ หรือโดยหน้าที่ฝังเดโมนี้ไว้)"],
    ["No camera was found\\.", "ไม่พบกล้อง"],
    ["The camera could not be opened \\((\\w+)\\)\\.", "เปิดกล้องไม่สำเร็จ ($1)"],
    ["This browser can't open a camera on this page \\(it needs a secure https:// address\\)\\.",
      "เบราว์เซอร์นี้เปิดกล้องในหน้านี้ไม่ได้ (ต้องใช้ที่อยู่ https:// ที่ปลอดภัย)"]
  ];
  var SIM = " Showing live mode on a moving example page instead, as if it were the camera\\.$";
  var SIM_TH = " จึงแสดงโหมดสดบนหน้าตัวอย่างที่เคลื่อนไหวแทน เสมือนว่าเป็นภาพจากกล้อง";
  var NEEDS = " Live mode needs a camera\\.$";
  var NEEDS_TH = " โหมดสดต้องใช้กล้อง";


  var patterns = [
    // status line: loading and reading errors (the error text itself comes from the browser)
    ["^Could not load the model: (.+)\\. Open this page through a web server, or use the single-file build\\.$",
      "โหลดโมเดลไม่สำเร็จ: $1 เปิดหน้านี้ผ่านเว็บเซิร์ฟเวอร์ หรือใช้เวอร์ชันไฟล์เดียว"],
    ["^Could not load the model: (.+)\\.$", "โหลดโมเดลไม่สำเร็จ: $1"],
    ["^Reading failed: (.+)$", "อ่านไม่สำเร็จ: $1"],

    // model info in the footer, one line per model set (model names are file names: kept)
    ["^Fast models: line detector (\\S+) \\(([\\d,]+) KB, ([\\d.]+) G MACs at (\\d+)×(\\d+)\\), horizontal reader (\\S+) \\(([\\d,]+) KB\\), vertical reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "โมเดลชุดเร็ว: ตัวตรวจจับบรรทัด $1 ($2 KB, $3 G MACs ที่ $4×$5), ตัวอ่านแนวนอน $6 ($7 KB), ตัวอ่านแนวตั้ง $8 ($9 KB); ตัวอ่านใช้ $10 M MACs ที่ $11×$12"],
    ["^Fast models: horizontal reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "โมเดลชุดเร็ว: ตัวอ่านแนวนอน $1 ($2 KB); ตัวอ่านใช้ $3 M MACs ที่ $4×$5"],
    ["^Accurate models: line detector (\\S+) \\(([\\d,]+) KB, ([\\d.]+) G MACs at (\\d+)×(\\d+)\\), horizontal reader (\\S+) \\(([\\d,]+) KB\\), vertical reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "โมเดลชุดแม่นยำ: ตัวตรวจจับบรรทัด $1 ($2 KB, $3 G MACs ที่ $4×$5), ตัวอ่านแนวนอน $6 ($7 KB), ตัวอ่านแนวตั้ง $8 ($9 KB); ตัวอ่านใช้ $10 M MACs ที่ $11×$12"],
    ["^Accurate models: horizontal reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "โมเดลชุดแม่นยำ: ตัวอ่านแนวนอน $1 ($2 KB); ตัวอ่านใช้ $3 M MACs ที่ $4×$5"],

    // example buttons
    ["^Example page (\\d+)$", "หน้าตัวอย่างที่ $1"],
    ["^Example line (\\d+)$", "บรรทัดตัวอย่างที่ $1"],

    // page summary
    ["^(\\d+) lines? found in (\\d+) ms, read in (\\d+) ms \\(([\\d.]+) ms per line\\)\\.$",
      "พบ $1 บรรทัดใน $2 ms อ่านเสร็จใน $3 ms ($4 ms ต่อบรรทัด)"],
    ["^No Hunter script found \\((\\d+) ms\\)\\. Tap the text to look closer, or drag a box around one line\\.$",
      "ไม่พบอักษรฮันเตอร์ ($1 ms) แตะที่ข้อความเพื่อให้มองใกล้ขึ้น หรือลากกรอบล้อมบรรทัดหนึ่ง"],
    ["^No Hunter script found \\((\\d+) ms\\)\\.$", "ไม่พบอักษรฮันเตอร์ ($1 ms)"],

    // line tags in any other combination
    ["^not Hunter script\\? · (.+)$", "ไม่ใช่อักษรฮันเตอร์? · $1"],

    // an unsure character: aria-label and tooltip (the kana and percentages are kept)
    ["^Unsure: (.+?) (\\d+%), maybe (.+?) \\((\\d+%)\\)$", "ไม่แน่ใจ: $1 $2 หรืออาจเป็น $3 ($4)"],
    ["^Unsure: (.+?) (\\d+%)$", "ไม่แน่ใจ: $1 $2"],
    ["^(\\S+) (\\d+%), maybe (\\S+) \\((\\d+%)\\)$", "$1 $2 หรืออาจเป็น $3 ($4)"],
    ["^(\\d+%) sure$", "มั่นใจ $1"],

    // the strip caption under the selected line
    ["^What the vertical reader saw: the column turned on its side, as a (\\d+) × (\\d+) px grayscale strip\\.$",
      "สิ่งที่ตัวอ่านแนวตั้งเห็น: คอลัมน์ที่ถูกหมุนให้นอนตะแคง เป็นแถบภาพเทาขนาด $1 × $2 px"],
    ["^What the horizontal reader saw: the column turned on its side, as a (\\d+) × (\\d+) px grayscale strip\\.$",
      "สิ่งที่ตัวอ่านแนวนอนเห็น: คอลัมน์ที่ถูกหมุนให้นอนตะแคง เป็นแถบภาพเทาขนาด $1 × $2 px"],
    ["^What the horizontal reader saw: the line as a (\\d+) × (\\d+) px grayscale strip\\.$",
      "สิ่งที่ตัวอ่านแนวนอนเห็น: บรรทัดนี้ในรูปแถบภาพเทาขนาด $1 × $2 px"],

    // notices under the selected line
    ["^This line is (\\d+) px long at (\\d+) px height, more than the reader's (\\d+) px, so it was squeezed ([\\d.]+)× to fit\\. Expect errors: drag boxes around about (\\d+) glyphs at a time\\.$",
      "บรรทัดนี้ยาว $1 px ที่ความสูง $2 px ซึ่งเกิน $3 px ที่ตัวอ่านรับได้ จึงถูกบีบลง $4× ให้พอดี อาจอ่านผิดได้ ให้ลากกรอบล้อมทีละราว $5 ตัวอักษร"],
    ["^The reader is unsure of this line \\((\\d+)% on average\\): it may not be Hunter script at all \\(Latin or Japanese text, art\\)\\.$",
      "ตัวอ่านไม่แน่ใจกับบรรทัดนี้ (เฉลี่ย $1%) บรรทัดนี้อาจไม่ใช่อักษรฮันเตอร์เลย (อาจเป็นตัวอักษรละตินหรือญี่ปุ่น หรือภาพวาด)"],
    ["^Line (\\d+) of (\\d+) · WebAssembly( SIMD)?$", "บรรทัดที่ $1 จาก $2 · WebAssembly$3"],

    // live camera: speed line
    ["^([\\d.]+) frames/s · detect (\\d+) ms · read (\\d+) ms \\((\\d+) kept\\)$", "$1 เฟรม/วินาที · ตรวจจับ $2 ms · อ่าน $3 ms (ใช้ผลเดิม $4)"],
    ["^([\\d.]+) frames/s · detect (\\d+) ms · read (\\d+) ms$", "$1 เฟรม/วินาที · ตรวจจับ $2 ms · อ่าน $3 ms"],
    ["^Showing live mode on a moving example page instead, as if it were the camera\\.$",
      "แสดงโหมดสดบนหน้าตัวอย่างที่เคลื่อนไหวแทน เสมือนว่าเป็นภาพจากกล้อง"]
  ];

  // live camera: why there is no camera, alone, with the simulated-camera note, or in the status line
  WHY.forEach(function (w) {
    patterns.push(["^" + w[0] + SIM, w[1] + SIM_TH]);
    patterns.push(["^" + w[0] + NEEDS, w[1] + NEEDS_TH]);
    patterns.push(["^" + w[0] + "$", w[1]]);
  });

  window.HXH_I18N = { strings: strings, patterns: patterns };
})();
