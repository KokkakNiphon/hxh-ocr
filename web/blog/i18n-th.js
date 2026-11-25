// Thai translations of the text that blog.js, explain.js and deep.js draw inside <article>
// (chart titles, labels, legends, SVG text, readouts, aria-labels). Loaded as a classic script
// before the modules; blog.js startTranslator() matches the trimmed text of a text node or
// attribute exactly against `strings`, else against the first matching `patterns` entry.
(function () {
  // the CTC figure (blog.js) appears on both pages: plain words in the explainer, the deep dive's
  // technical register (สเต็ป, blank) in the research article
  var DEEP = /deep-dive/.test(location.pathname);
  var reg = function (plain, deep) { return DEEP ? deep : plain; };
  var strings = {
    // ------------------------------------------------------------ shared controls
    "Pause": "หยุดชั่วคราว",
    "Play": "เล่น",

    // ------------------------------------------------------------ blog.js: glyph chart and encoder
    "The long-vowel mark (chōon): it lengthens the vowel before it, so はんたー reads hantā.": "เครื่องหมายเสียงยาว (โชอง): ทำให้สระที่อยู่ข้างหน้ายาวขึ้น はんたー จึงอ่านว่า hantā",
    "wide 50% · narrow (\"4\") 50%": "แบบกว้าง 50% · แบบแคบ (\"4\") 50%",
    "left-pointing (Ƹ) 85% · right-pointing (Σ) 15%": "หันซ้าย (Ƹ) 85% · หันขวา (Σ) 15%",
    "filled dot 50% · hollow ring 50%": "จุดทึบ 50% · วงกลมกลวง 50%",

    // ------------------------------------------------------------ blog.js: CTC figure (both pages)
    "Reading one line, step by step": reg("อ่านหนึ่งบรรทัด ทีละก้าว", "การอ่านหนึ่งบรรทัดทีละสเต็ป"),
    "Back to the start": reg("กลับไปจุดเริ่มต้น", "กลับไปสเต็ปแรก"),
    "One step back": reg("ถอยหนึ่งก้าว", "ถอยหนึ่งสเต็ป"),
    "One step forward": reg("ไปข้างหน้าหนึ่งก้าว", "ไปข้างหน้าหนึ่งสเต็ป"),
    "Step": reg("ก้าว", "สเต็ป"),
    "Slow": "ช้า",
    "Close-up: the steps around the cursor": reg("ภาพขยาย: ก้าวรอบ ๆ เคอร์เซอร์", "ภาพขยาย: สเต็ปรอบเคอร์เซอร์"),
    "Close-up of the steps around the cursor": reg("ภาพขยายของก้าวรอบ ๆ เคอร์เซอร์", "ภาพขยายของสเต็ปรอบเคอร์เซอร์"),
    "Top guesses at this step": reg("คำตอบที่เดาไว้อันดับต้น ๆ ที่ก้าวนี้", "คำตอบอันดับต้นที่สเต็ปนี้"),
    "1 · What each step says": reg("1 · แต่ละก้าวบอกว่าอะไร", "1 · ผลของแต่ละสเต็ป"),
    "2 · Glue repeats together": reg("2 · รวมตัวที่ซ้ำกัน", "2 · รวมตัวซ้ำที่ติดกัน"),
    "3 · Drop the blanks": reg("3 · ทิ้งก้าวที่ไม่มีอะไร", "3 · ตัดตัวว่างทิ้ง"),
    "nothing": reg("ไม่มีอะไร", "ว่าง"),
    "space": "เว้นวรรค",
    "blank": "ว่าง",

    // ------------------------------------------------------------ blog.js: pipeline flow (deep dive)
    "Page": "หน้า",
    "any size · gray": "ขนาดใดก็ได้ · ภาพเทา",
    "Any photo, screenshot or scan, turned gray in the browser.": "รูปถ่าย ภาพแคปหน้าจอ หรือภาพสแกนใดก็ได้ แปลงเป็นภาพระดับเทาในเบราว์เซอร์",
    "Letterbox": "เลตเตอร์บ็อกซ์",
    "Shrunk so the long side is 640 px, placed top-left, padded white. Rust, area averaging.": "ย่อให้ด้านยาวเป็น 640 px วางชิดมุมซ้ายบน เติมขอบสีขาว ใช้ Rust ย่อแบบเฉลี่ยพื้นที่",
    "Detector": "ตัวตรวจจับ",
    "Two maps at 320 × 320: where each line's shrunken core is, and whether it runs across or down.": "แผนที่สองแผ่นขนาด 320 × 320: ตำแหน่งแกนที่หดแล้วของแต่ละบรรทัด และบรรทัดนั้นเป็นแนวนอนหรือแนวตั้ง",
    "Lines": "บรรทัด",
    "Connected cores → minimum-area rectangles → grown back (unclip 1.5) → reading order, columns right to left.": "แกนที่เชื่อมต่อกัน → สี่เหลี่ยมพื้นที่น้อยที่สุด → ขยายกลับ (unclip 1.5) → ลำดับการอ่าน คอลัมน์จากขวาไปซ้าย",
    "Crops": "ภาพตัด",
    "48 × 768 strips": "แถบ 48 × 768",
    "Each line cut out straight with margins at the page's own resolution; columns turned a quarter turn.": "ตัดแต่ละบรรทัดออกมาให้ตรงพร้อมระยะขอบ ที่ความละเอียดเดิมของหน้า คอลัมน์ถูกหมุนหนึ่งในสี่รอบ",
    "Readers": "ตัวอ่าน",
    "int8 · 154 M MACs per line": "int8 · 154 M MACs ต่อบรรทัด",
    "A horizontal reader and a vertical reader turn each strip into kana; unsure lines are set aside.": "ตัวอ่านแนวนอนและตัวอ่านแนวตั้งแปลงแต่ละแถบเป็นคานะ บรรทัดที่ไม่มั่นใจจะถูกพักไว้",

    // ------------------------------------------------------------ blog.js: bar charts (deep dive)
    "Character error rate on each validation set": "อัตราความผิดพลาดระดับอักขระบนชุดตรวจสอบแต่ละชุด",
    "48 px reader, fp32, 3,000 lines per set (lower is better)": "ตัวอ่าน 48 px, fp32, ชุดละ 3,000 บรรทัด (ยิ่งต่ำยิ่งดี)",
    "v3 val (clean)": "v3 val (สะอาด)",
    "v4 val (official forms, art, bold)": "v4 val (รูปแบบทางการ, ภาพประกอบ, ตัวหนา)",
    "trained on v3": "ฝึกด้วย v3",
    "trained on v4": "ฝึกด้วย v4",
    "Milliseconds per inference": "มิลลิวินาทีต่อการอนุมานหนึ่งครั้ง",
    "node 24, one thread of a Ryzen 7 3700X (lower is better)": "node 24, หนึ่งเธรดของ Ryzen 7 3700X (ยิ่งต่ำยิ่งดี)",
    "scalar": "สเกลาร์",
    "48 px reader": "ตัวอ่าน 48 px",
    "32 px reader": "ตัวอ่าน 32 px",
    "three-stage reader": "ตัวอ่านสามสเตจ",
    "Detector recall by line thickness": "ความครบถ้วน (recall) ของตัวตรวจจับตามความหนาบรรทัด",
    "share of ground-truth lines found at IoU ≥ 0.5; thickness in the 640 px letterbox": "สัดส่วนของบรรทัดค่าจริงที่หาเจอที่ IoU ≥ 0.5; ความหนาวัดในเลตเตอร์บ็อกซ์ 640 px",

    // ------------------------------------------------------------ explain.js: decoder, three steps, maker
    "(long)": "(เสียงยาว)",
    "hantā raisensu  →  “Hunter License”": "hantā raisensu  →  “ใบอนุญาตฮันเตอร์”",
    "A training picture being made": "ภาพฝึกกำลังถูกสร้างขึ้น",
    "Previous step": "ขั้นก่อนหน้า",
    "Next step": "ขั้นถัดไป",
    "Draw the symbols": "วาดสัญลักษณ์",
    "Clean strokes from the alphabet: ご (go) is こ plus the voicing dot, then ん (n).": "เส้นเรียบร้อยจากชุดตัวอักษร: ご (go) คือ こ บวกจุดเสียงก้อง ตามด้วย ん (n)",
    "Draw them by hand": "วาดให้เหมือนเขียนมือ",
    "Every point is nudged a little and the pen gets thicker, like lettering drawn by hand.": "ทุกจุดถูกขยับเล็กน้อยและเส้นหนาขึ้น เหมือนตัวหนังสือที่เขียนด้วยมือ",
    "Put them somewhere": "วางลงบนพื้นหลัง",
    "A random background: here the dotted screentone used for shading in manga.": "สุ่มพื้นหลังขึ้นมา ในที่นี้คือสกรีนโทนลายจุดที่ใช้ลงเงาในมังงะ",
    "Tilt it": "ทำให้เอียง",
    "A slight slant and squash, as if photographed at an angle.": "เอียงและบีบเล็กน้อย เหมือนถ่ายจากมุมเฉียง",
    "Blur and add grain": "ทำให้เบลอและใส่เกรน",
    "Out of focus, a little noisy, slightly faded: like a phone photo of a page.": "หลุดโฟกัส มีจุดรบกวนนิดหน่อย สีซีดลงเล็กน้อย เหมือนถ่ายหน้ากระดาษด้วยโทรศัพท์",
    "Step 1: Draw the symbols": "ขั้นที่ 1: วาดสัญลักษณ์",
    "Step 2: Draw them by hand": "ขั้นที่ 2: วาดให้เหมือนเขียนมือ",
    "Step 3: Put them somewhere": "ขั้นที่ 3: วางลงบนพื้นหลัง",
    "Step 4: Tilt it": "ขั้นที่ 4: ทำให้เอียง",
    "Step 5: Blur and add grain": "ขั้นที่ 5: ทำให้เบลอและใส่เกรน",

    // ------------------------------------------------------------ explain.js: one number two ways, timings
    "1 byte": "1 ไบต์",
    "4 bytes": "4 ไบต์",
    "what models are usually trained with": "แบบที่มักใช้ตอนฝึกโมเดล",
    "what runs on your phone: close enough, four times smaller": "แบบที่รันบนโทรศัพท์ของคุณ: ใกล้เคียงพอ แต่เล็กกว่าสี่เท่า",
    "Time for each step": "เวลาของแต่ละขั้นตอน",
    "find every line": "หาทุกบรรทัด",
    "(line finder)": "(ตัวหาบรรทัด)",
    "read one line": "อ่านหนึ่งบรรทัด",
    "(reader)": "(ตัวอ่าน)",
    "the whole page,": "ทั้งหน้า",
    "every line read": "อ่านครบทุกบรรทัด",

    // ------------------------------------------------------------ explain.js: privacy, sideways, vote
    "Diagram: the photo, the line finder, the reader and the text all stay inside the phone; nothing goes to the internet.": "แผนภาพ: รูปถ่าย ตัวหาบรรทัด ตัวอ่าน และข้อความ อยู่ภายในโทรศัพท์ทั้งหมด ไม่มีอะไรถูกส่งออกไปยังอินเทอร์เน็ต",
    "photo": "รูปถ่าย",
    "your picture": "รูปของคุณ",
    "find": "หา",
    "line finder": "ตัวหาบรรทัด",
    "read": "อ่าน",
    "two readers": "ตัวอ่าน 2 ตัว",
    "inside your browser": "ภายในเบราว์เซอร์ของคุณ",
    "no upload": "ไม่มีการอัปโหลด",
    "A vertical column of four glyphs turns a quarter turn to the left and becomes a horizontal strip.": "คอลัมน์แนวตั้งที่มีสัญลักษณ์สี่ตัวหมุนไปทางซ้ายหนึ่งในสี่รอบ แล้วกลายเป็นแถบแนวนอน",
    "on the page: top to bottom": "บนหน้ากระดาษ: จากบนลงล่าง",
    "the vertical reader sees: はんたー": "ตัวอ่านแนวตั้งเห็นเป็น: はんたー",
    "label shown": "ข้อความที่แสดง",
    "illustration": "ภาพประกอบ",

    // ------------------------------------------------------------ explain.js: the big picture (journey)
    "Photo": "รูปถ่าย",
    "You take a picture.": "คุณถ่ายรูป",
    "Numbers": "ตัวเลข",
    "The computer sees a grid of numbers.": "คอมพิวเตอร์เห็นตารางตัวเลข",
    "Find lines": "หาบรรทัด",
    "The line finder spots every line of writing.": "ตัวหาบรรทัดมองหาตัวหนังสือทุกบรรทัด",
    "Cut out": "ตัดออกมา",
    "Each line is cut out and straightened.": "แต่ละบรรทัดถูกตัดออกมาและจัดให้ตรง",
    "Read": "อ่าน",
    "The reader turns each strip into kana.": "ตัวอ่านแปลงแต่ละแถบเป็นคานะ",
    "Words": "ถ้อยคำ",
    "Kana become Latin letters you can read.": "คานะกลายเป็นตัวอักษรละตินที่คุณอ่านได้",

    // ------------------------------------------------------------ explain.js: what a computer sees (pixels)
    "A 22 by 22 grid of pixels showing one symbol": "ตารางพิกเซล 22 × 22 ที่แสดงสัญลักษณ์หนึ่งตัว",
    "Show the numbers": "แสดงตัวเลข",
    "Hide the numbers": "ซ่อนตัวเลข",
    "Point at a pixel": "ชี้ที่พิกเซลใดก็ได้",
    "0 = black": "0 = ดำ",
    "128 = gray": "128 = เทา",
    "255 = white": "255 = ขาว",

    // ------------------------------------------------------------ explain.js: filters (conv)
    "Filter": "ฟิลเตอร์",
    "Picture (darkness 0–9)": "ภาพ (ความเข้ม 0–9)",
    "Result": "ผลลัพธ์",
    "Up-and-down edges": "ขอบแนวขึ้นลง",
    "Across edges": "ขอบแนวขวาง",
    "Blur": "เบลอ",
    "big where dark meets light from left to right": "ค่ามากตรงที่สีเข้มเจอสีอ่อนจากซ้ายไปขวา",
    "big where dark meets light from top to bottom": "ค่ามากตรงที่สีเข้มเจอสีอ่อนจากบนลงล่าง",
    "adds up the neighbourhood: softens everything": "รวมค่าของพิกเซลรอบ ๆ: ทำให้ทุกอย่างนุ่มลง",

    // ------------------------------------------------------------ explain.js: layers
    "Pixels": "พิกเซล",
    "the input": "ข้อมูลที่ป้อนเข้า",
    "Edges": "ขอบ",
    "early layers": "ชั้นแรก ๆ",
    "Strokes": "เส้น",
    "middle layers": "ชั้นกลาง ๆ",
    "Parts": "ส่วนประกอบ",
    "later layers": "ชั้นหลัง ๆ",
    "the last layer": "ชั้นสุดท้าย",

    // ------------------------------------------------------------ explain.js: guess, check, adjust (learn)
    "The practice loop: show, guess, check, adjust": "วงจรการฝึกซ้อม: ให้ดู เดา ตรวจ ปรับ",
    "Show": "ให้ดู",
    "examples": "ตัวอย่าง",
    "Guess": "เดา",
    "what they say": "ว่าคืออะไร",
    "Check": "ตรวจ",
    "against answers": "กับคำตอบ",
    "Adjust": "ปรับ",
    "the numbers": "ตัวเลข",
    "practice rounds": "รอบการฝึก",
    "How many symbols the reader got wrong while it trained": "จำนวนสัญลักษณ์ที่ตัวอ่านอ่านผิดระหว่างการฝึก",
    "symbols wrong per 100": "สัญลักษณ์ที่ผิดต่อ 100 ตัว",

    // ------------------------------------------------------------ nn.js: a network you can poke (tinynet)
    "A real neural network, small enough to see": "โครงข่ายประสาทเทียมของจริง ที่เล็กพอจะมองเห็นได้ทั้งหมด",
    "Symbol to draw": "สัญลักษณ์ที่จะวาด",
    "Wobble it": "ทำให้เบี้ยว",
    "Clear": "ล้าง",
    "The grid is empty: paint something.": "ตารางว่างเปล่า ลองระบายอะไรสักอย่างดู",
    "Try this: pick ぐ, then erase its little dot. Then hover over (or tap) a neuron in the first layer to see which pixels it looks for.": "ลองดู: เลือก ぐ แล้วลบจุดเล็ก ๆ ของมันออก จากนั้นชี้ (หรือแตะ) ที่นิวรอนในชั้นแรก เพื่อดูว่ามันมองหาพิกเซลไหน",

    // ------------------------------------------------------------ nn.js: inside the real reader
    "Inside the real reader": "ข้างในตัวอ่านของจริง",
    "Input": "อินพุต",
    "One list per step": "หนึ่งรายการต่อก้าว",
    "Scores": "คะแนน",

    // ------------------------------------------------------------ deep.js: system overview
    "Offline · Python, PyTorch and Rust on one PC": "ออฟไลน์ · Python, PyTorch และ Rust บนพีซีเครื่องเดียว",
    "Online · the browser, Rust compiled to WebAssembly": "ออนไลน์ · เบราว์เซอร์ และ Rust ที่คอมไพล์เป็น WebAssembly",
    "Glyph strokes": "เส้นของกลิฟ",
    "47 traced glyphs, 3 with two forms": "กลิฟที่ลอกลาย 47 ตัว 3 ตัวมีสองรูปแบบ",
    "Generators": "ตัวสร้างข้อมูล",
    "lines (Python), pages (Rust)": "บรรทัด (Python), หน้า (Rust)",
    "Train fp32": "ฝึกแบบ fp32",
    "PyTorch; EMA + BN re-estimation": "PyTorch; EMA + ประมาณ BN ใหม่",
    "ESP-DL-exact fake quantisation": "ควอนไทซ์จำลองตรงตาม ESP-DL",
    "int8 .espdl, power-of-two scales": "int8 .espdl, สเกลกำลังของสอง",
    "converter, checked per filter": "ตัวแปลง ตรวจทุกฟิลเตอร์",
    "Image": "ภาพ",
    "photo, screenshot or camera": "รูปถ่าย ภาพแคปจอ หรือกล้อง",
    "gray, 640 × 640": "ภาพเทา, 640 × 640",
    "Post-processing": "ประมวลผลภายหลัง",
    "lines, order, crops": "บรรทัด ลำดับ ภาพตัด",
    "Readers h / v": "ตัวอ่าน h / v",
    "int8, 154 M MACs per line": "int8, 154 M MACs ต่อบรรทัด",
    "Greedy CTC": "CTC แบบละโมบ",
    "kana, romaji, confidence": "คานะ โรมาจิ ความมั่นใจ",

    // ------------------------------------------------------------ deep.js: glyph forms and marks
    "ま: left-pointing (Ƹ), right-pointing (Σ)": "ま: หันซ้าย (Ƹ), หันขวา (Σ)",
    "せ: wide, narrow (\"4\")": "せ: แบบกว้าง, แบบแคบ (\"4\")",
    "も: filled dot, hollow ring": "も: จุดทึบ, วงกลมกลวง",
    "left-pointing (Ƹ) · right-pointing (Σ)": "หันซ้าย (Ƹ) · หันขวา (Σ)",
    "wide · narrow (\"4\")": "แบบกว้าง · แบบแคบ (\"4\")",
    "filled dot · hollow ring": "จุดทึบ · วงกลมกลวง",
    "Where the generator places voicing marks": "ตำแหน่งที่ตัวสร้างข้อมูลวางเครื่องหมายเสียงก้อง",
    "bbox of the drawn form": "กรอบล้อมรอบของรูปแบบที่วาด",
    "dot centres, v4:": "จุดศูนย์กลางของจุด, v4:",

    // ------------------------------------------------------------ deep.js: turned glyphs
    "ふ turned a quarter turn counter-clockwise looks like う": "ふ ที่หมุนทวนเข็มนาฬิกาหนึ่งในสี่รอบดูเหมือน う",
    "ふ upright": "ふ ตั้งตรง",
    "90° ccw": "90° ทวนเข็ม",
    "ふ turned": "ふ ที่หมุนแล้ว",
    "う upright": "う ตั้งตรง",
    "A shared reader would have to be told the orientation; a separate vertical reader avoids the collision.": "ตัวอ่านที่ใช้ร่วมกันจะต้องได้รับการบอกทิศทาง การแยกตัวอ่านแนวตั้งออกมาช่วยเลี่ยงการชนกันนี้",

    // ------------------------------------------------------------ deep.js: keep-clear rule
    "Keep-clear rule between labelled Hunter lines and foreign text": "กฎเว้นระยะปลอดภัยระหว่างบรรทัดฮันเตอร์ที่ติดป้ายกำกับกับข้อความอื่น",
    "labelled Hunter quad, height h₁": "รูปสี่เหลี่ยมฮันเตอร์ที่ติดป้ายกำกับ สูง h₁",
    "overlaps the margin: rejected ✗": "ทับระยะขอบ: ไม่วาง ✗",
    "foreign run, height h₂: placed ✓": "ช่วงข้อความอื่น สูง h₂: วางได้ ✓",
    "margin": "ระยะขอบ",
    "margin = 0.3 × max(h₁, h₂) + the foreign run's outline or shadow halo": "ระยะขอบ = 0.3 × max(h₁, h₂) + เส้นขอบหรือรัศมีเงาของช่วงข้อความอื่น",

    // ------------------------------------------------------------ deep.js: reader and detector architecture
    "Reader architecture with tensor shapes": "สถาปัตยกรรมตัวอ่านพร้อมรูปร่างเทนเซอร์",
    "input": "อินพุต",
    "stem": "สเต็ม",
    "mean H": "เฉลี่ยตาม H",
    "head": "ส่วนหัว",
    "classes": "คลาส",
    "gray": "ภาพเทา",
    "2 convs": "conv 2 ชั้น",
    "6 UIB": "UIB 6 บล็อก",
    "avg pool": "พูลเฉลี่ย",
    "2 × res.": "res. 2 บล็อก",
    "T = 192 columns → greedy CTC": "T = 192 คอลัมน์ → CTC แบบละโมบ",
    "DBNet-lite detector architecture": "สถาปัตยกรรมตัวตรวจจับ DBNet-lite",
    "backbone": "แบ็กโบน",
    "stem 3×3 s2": "สเต็ม 3×3 s2",
    "→ 16 ch": "→ 16 แชนเนล",
    "concat": "ต่อกัน",
    "nearest up ×2": "ขยาย nearest ×2",
    "ch 0: text-kernel logit": "ช่อง 0: logit เคอร์เนลข้อความ",
    "ch 1: vertical logit": "ช่อง 1: logit แนวตั้ง",
    "threshold head": "ส่วนหัวค่าเกณฑ์",
    "training only": "ใช้ตอนฝึกเท่านั้น",
    "backbone 442 M MACs": "แบ็กโบน 442 M MACs",
    "head 947 M": "ส่วนหัว 947 M",

    // ------------------------------------------------------------ deep.js: target geometry
    "length / thickness": "ความยาว / ความหนา",
    "Line length as a multiple of its thickness": "ความยาวบรรทัดเป็นกี่เท่าของความหนา",
    "Shrink, threshold band and unclip of one line": "การหด แถบค่าเกณฑ์ และการ unclip ของหนึ่งบรรทัด",
    "ground truth = exact inverse": "ค่าจริง = ตัวผกผันแม่นตรง",
    "kernel": "เคอร์เนล",
    "threshold band ±D": "แถบค่าเกณฑ์ ±D",

    // ------------------------------------------------------------ deep.js: int8 grids, requantisation, exponents
    "tensor": "เทนเซอร์",
    "Quantised tensor of the 48 px reader": "เทนเซอร์ที่ถูกควอนไทซ์ของตัวอ่าน 48 px",
    "Integer path of one convolution": "เส้นทางจำนวนเต็มของคอนโวลูชันหนึ่งชั้น",
    "exponents": "เลขชี้กำลัง",
    "Σ x·w + bias": "Σ x·w + ไบแอส",
    "int32 at": "int32 ที่",
    "shift s": "เลื่อนบิต s",
    "round half up": "ปัดครึ่งขึ้น",
    "saturate": "ตัดค่าอิ่มตัว",
    "clamp to": "จำกัดค่าใน",
    "if fused": "ถ้ารวมไว้",
    "exponent": "เลขชี้กำลัง",
    "exponents e_x, e_w": "เลขชี้กำลัง e_x, e_w",
    "int32 at e_x + e_w": "int32 ที่ e_x + e_w",
    "clamp to [−128, 127]": "จำกัดค่าใน [−128, 127]",
    "exponent e_y": "เลขชี้กำลัง e_y",
    "3×3 conv weights": "น้ำหนัก conv 3×3",
    "depthwise weights": "น้ำหนัก depthwise",
    "1×1 weights": "น้ำหนัก 1×1",
    "output activation exponent": "เลขชี้กำลังของแอกทิเวชันเอาต์พุต",
    "Weight and output exponents of every convolution": "เลขชี้กำลังของน้ำหนักและเอาต์พุตของทุกคอนโวลูชัน",
    "stem–2": "สเต็ม–2",
    "convolutions in network order →": "คอนโวลูชันตามลำดับในเครือข่าย →",
    "exponent e": "เลขชี้กำลัง e",

    // ------------------------------------------------------------ deep.js: QAT and training curves
    "model": "โมเดล",
    "48 px horizontal": "48 px แนวนอน",
    "48 px vertical": "48 px แนวตั้ง",
    "32 px three-stage": "32 px สามสเตจ",
    "validation CER on 2,000 lines; synth_v4; lower is better": "CER บนชุดตรวจสอบ 2,000 บรรทัด; synth_v4; ยิ่งต่ำยิ่งดี",
    "int8 (fake-quantised)": "int8 (ควอนไทซ์จำลอง)",
    "same weights in float": "น้ำหนักเดียวกันแบบ float",
    "round": "รอบ",
    "Round 1 (synth_v1 + v2, 32 px): pretrained vs from scratch": "รอบที่ 1 (synth_v1 + v2, 32 px): ฝึกล่วงหน้าเทียบกับฝึกจากศูนย์",
    "Round 2 (synth_v3): 32 px vs 48 px vs 3 stages": "รอบที่ 2 (synth_v3): 32 px เทียบ 48 px เทียบ 3 สเตจ",
    "Line readers on synth_v4: validation CER while training (fp32)": "ตัวอ่านบรรทัดบน synth_v4: CER บนชุดตรวจสอบระหว่างการฝึก (fp32)",
    "val CER (%) on a fixed 2,000-line subset, log scale; lower is better": "val CER (%) บนชุดย่อยคงที่ 2,000 บรรทัด สเกลลอการิทึม; ยิ่งต่ำยิ่งดี",
    "MobileNetV4-small, ImageNet-pretrained": "MobileNetV4-small, ฝึกล่วงหน้าด้วย ImageNet",
    "MobileNetV4-small, from scratch": "MobileNetV4-small, ฝึกจากศูนย์",
    "tiny CNN, from scratch": "CNN ขนาดจิ๋ว, ฝึกจากศูนย์",
    "3-stage MobileNetV4-small, 32 px": "MobileNetV4-small 3 สเตจ, 32 px",
    "MobileNetV4-small, 48 px, horizontal": "MobileNetV4-small, 48 px, แนวนอน",
    "MobileNetV4-small, 48 px, vertical columns": "MobileNetV4-small, 48 px, คอลัมน์แนวตั้ง",
    "3-stage MobileNetV4-small, 32 px, horizontal": "MobileNetV4-small 3 สเตจ, 32 px, แนวนอน",

    // ------------------------------------------------------------ deep.js: detector curve, confidence filter
    "det_v1: validation detection while training": "det_v1: ผลการตรวจจับบนชุดตรวจสอบระหว่างการฝึก",
    "300 pages_v2 val pages, quad IoU ≥ 0.5, every 2,000 steps; higher is better": "หน้า pages_v2 val 300 หน้า, IoU ของรูปสี่เหลี่ยม ≥ 0.5, ทุก 2,000 สเต็ป; ยิ่งสูงยิ่งดี",
    "precision": "ความแม่นยำ",
    "recall": "ความครบถ้วน",
    "Detection with a minimum line confidence": "การตรวจจับเมื่อกำหนดความมั่นใจขั้นต่ำของบรรทัด",
    "precision, recall and F1 (%)": "ความแม่นยำ ความครบถ้วน และ F1 (%)",
    "none": "ไม่มี",
    "Character error rates": "อัตราความผิดพลาดระดับอักขระ",
    "end-to-end CER counts hidden lines as missed; matched-line CER does not": "CER ตั้งแต่ต้นจนจบนับบรรทัดที่ถูกซ่อนเป็นบรรทัดที่หาไม่เจอ แต่ CER ของบรรทัดที่จับคู่ได้ไม่นับ",
    "CER h, end to end": "CER h, ตั้งแต่ต้นจนจบ",
    "CER v, end to end": "CER v, ตั้งแต่ต้นจนจบ",
    "CER of matched lines": "CER ของบรรทัดที่จับคู่ได้",
    "deployed": "ใช้งานจริง",

    // ------------------------------------------------------------ deep.js: page timings
    "letterbox": "เลตเตอร์บ็อกซ์",
    "detector (int8)": "ตัวตรวจจับ (int8)",
    "post-processing": "ประมวลผลภายหลัง",

    // ------------------------------------------------------------ deep.js: tap a line
    "Tap-a-line windows on a 3,000 px photo": "หน้าต่างของการแตะบรรทัดบนภาพถ่าย 3,000 px",
    "photo 3,000 × 3,000 px": "ภาพถ่าย 3,000 × 3,000 px",
    "Page pass": "รอบทั้งหน้า",
    "whole photo shrunk to 640 px (× 0.21)": "ย่อทั้งภาพเหลือ 640 px (× 0.21)",
    "found 4 of the page's 14 lines": "หาเจอ 4 จาก 14 บรรทัดของหน้า",
    "Tap": "แตะ",
    "windows around the tap, cut from the": "หน้าต่างรอบจุดที่แตะ ตัดจากภาพ",
    "original: 640 px first (no shrinking),": "ต้นฉบับ: 640 px ก่อน (ไม่ย่อเลย)",
    "then 1,280 px, then 320 px": "จากนั้น 1,280 px แล้ว 320 px",
    "read 13 of 14 lines exactly": "อ่านถูกทั้งหมด 13 จาก 14 บรรทัด",
    "75–110 ms per tap": "75–110 ms ต่อการแตะ",
    "The first window with a line under the tap": "หน้าต่างแรกที่มีบรรทัดอยู่ใต้จุดที่แตะ",
    "(or within 1.5 line thicknesses) wins.": "(หรือห่างไม่เกิน 1.5 เท่าความหนาบรรทัด) ชนะ",
    "schematic; window sizes to scale": "แผนภาพ; ขนาดหน้าต่างตามสัดส่วนจริง",

    // ------------------------------------------------------------ deep.js: false lines per page
    "Shown false lines per page, det_v1 vs det_v2": "บรรทัดลวงที่แสดงต่อหน้า, det_v1 เทียบ det_v2",
    "Shown false lines per page, det_v1": "บรรทัดลวงที่แสดงต่อหน้า, det_v1",
    "browser pipeline, unclip 1.5, margins 0.3 / 0.2; a false line is on text or icons if ≥ 40% of it overlaps them": "ไปป์ไลน์ในเบราว์เซอร์, unclip 1.5, ระยะขอบ 0.3 / 0.2; นับว่าบรรทัดลวงอยู่บนข้อความหรือไอคอนหากทับกัน ≥ 40%",
    "on ordinary text": "บนข้อความทั่วไป",
    "on icons": "บนไอคอน",
    "elsewhere": "ที่อื่น",
    "pages_foreign · 150 pages": "pages_foreign · 150 หน้า",
    "pages_v3 val · 100 mixed pages": "pages_v3 val · 100 หน้าแบบผสม",
    "det_v1, no filter": "det_v1, ไม่กรอง",
    "det_v1, min conf 0.85": "det_v1, กรองที่ 0.85",
    "det_v2, no filter": "det_v2, ไม่กรอง",
    "det_v2, min conf 0.8": "det_v2, กรองที่ 0.8",
    "no filter": "ไม่กรอง",
    "min conf 0.85": "กรองที่ 0.85",
    "min conf 0.9": "กรองที่ 0.9",

    // ------------------------------------------------------------ deep.js: live tracker
    "Track and vote across frames": "การติดตามและโหวตข้ามเฟรม",
    "frame": "เฟรม",
    "detector": "ตัวตรวจจับ",
    "reader": "ตัวอ่าน",
    "vote over last 5": "โหวต 5 เฟรมล่าสุด",
    "shown": "ที่แสดง",
    "box": "กรอบ",
    "reused": "ใช้ซ้ำ",
    "hidden": "ซ่อน",
    "misread": "อ่านผิด",
    "3 equal reads in a row → reuse for 2 frames": "อ่านได้เหมือนกัน 3 ครั้งติด → ใช้ซ้ำ 2 เฟรม",
    "at t6: はんたー ×4, はんだー ×1 → はんたー": "ที่ t6: はんたー ×4, はんだー ×1 → はんたー",
    "seen once": "เห็นครั้งเดียว",
    "schematic, readings illustrative": "แผนภาพ ผลการอ่านเป็นเพียงตัวอย่าง"
  };

  // the int8-grid readout and the per-conv exponent readout share long tails; build them once
  var GRID_TAIL = " · e = (-?\\d+) · step 2\\^(-?\\d+) = (\\S+) · int8 range (\\[[^\\]]+\\]) · KL clip c = (\\S+) · log2\\(c/128\\) = (\\S+) → (-?\\d+)$";
  var GRID_OUT = " · e = $1 · ขั้น 2^$2 = $3 · ช่วง int8 $4 · ค่าตัด KL c = $5 · log2(c/128) = $6 → $7";
  var EXP_TAIL = " · weights e_w = (-?\\d+) · input e_x = (-?\\d+) · output e_y = (-?\\d+) · shift s = (-?\\d+)";
  var EXP_OUT = " · น้ำหนัก e_w = $2 · อินพุต e_x = $3 · เอาต์พุต e_y = $4 · เลื่อน s = $5";
  var KINDS = [["full", "แบบเต็ม"], ["depthwise", "depthwise"], ["pointwise", "pointwise"]];

  var patterns = [
    // blog.js
    ["^Romaji “([^”]+)”\\. With a mark it also writes (.+)\\.$", "โรมาจิ “$1” เมื่อเติมเครื่องหมายจะเขียนเป็น $2 ได้ด้วย"],
    ["^Romaji “([^”]+)”\\.$", "โรมาจิ “$1”"],
    ["^(.+) in Hunter script$", "$1 ในอักษรฮันเตอร์"],
    ["^The real reader on (.+) \\((.+)\\)\\. It answers once for every 4 pixels of the strip: (\\d+) steps\\. The (\\d+) steps after the end of the line all say \"nothing here\" and are cut off\\.$",
      reg("ตัวอ่านของจริงกับ $1 ($2) มันตอบหนึ่งครั้งทุก ๆ 4 พิกเซลของแถบ รวม $3 ก้าว อีก $4 ก้าวที่อยู่หลังสุดบรรทัดบอกว่า \"ไม่มีอะไร\" ทั้งหมด จึงถูกตัดออก",
        "ตัวอ่านจริงบน $1 ($2) ให้คำตอบหนึ่งครั้งต่อทุก 4 พิกเซลของแถบ: $3 สเต็ป อีก $4 สเต็ปหลังจุดสิ้นสุดของบรรทัดล้วนเป็น \"ว่าง\" (blank) และถูกตัดออก")],
    ["^CTC decoding of (.+)$", "การถอดรหัส CTC ของ $1"],
    ["^step (\\d+) / (\\d+) · nothing here (\\S+) · read so far: (.*)$",
      reg("ก้าว $1 / $2 · ไม่มีอะไร $3 · อ่านได้แล้ว: $4", "สเต็ป $1 / $2 · ว่าง $3 · อ่านได้ถึงตอนนี้: $4")],
    ["^step (\\d+) / (\\d+) · (.+?) (\\S+) · read so far: (.*)$",
      reg("ก้าว $1 / $2 · $3 $4 · อ่านได้แล้ว: $5", "สเต็ป $1 / $2 · $3 $4 · อ่านได้ถึงตอนนี้: $5")],
    ["^(\\d+) × (\\d+) px · gray$", "$1 × $2 px · ภาพเทา"],
    ["^int8 · 154 M MACs per line · (.+) each$", "int8 · 154 M MACs ต่อบรรทัด · บรรทัดละ $1"],
    ["^Strip: (.+)$", "แถบ: $1"],
    ["^vertical · (\\d+)%$", "แนวตั้ง · $1%"],
    ["^horizontal · (\\d+)%$", "แนวนอน · $1%"],
    ["^(\\d+) lines shown on this page · about (\\d+) ms end to end in node$", "แสดง $1 บรรทัดในหน้านี้ · ใช้เวลาตั้งแต่ต้นจนจบราว $2 ms ใน node"],

    // explain.js
    ["^Straightened strip of line (\\d+)$", "แถบที่จัดให้ตรงแล้วของบรรทัดที่ $1"],
    ["^Step (\\d+) of (\\d+)$", "ขั้นที่ $1 จาก $2"],
    ["^Step (\\d+): (.+)$", "ขั้นที่ $1: $2"],
    ["^The example page in this article, (\\d+) lines; milliseconds on one core of a desktop processor \\(lower is faster\\)$",
      "หน้าตัวอย่างในบทความนี้ มี $1 บรรทัด หน่วยเป็นมิลลิวินาทีบนโปรเซสเซอร์เดสก์ท็อปหนึ่งคอร์ (ยิ่งน้อยยิ่งเร็ว)"],
    ["^look (\\d+)$", "ครั้งที่ $1"],
    ["^row (\\d+), column (\\d+): (\\d+)$", "แถว $1 คอลัมน์ $2: $3"],
    ["^(\\d+\\.\\d) wrong$", "ผิด $1"],

    // nn.js: tinynet
    ["^It knows only (\\d+) symbols\\. Pick one, or paint and erase pixels on the grid, and watch the signal flow through it\\.$",
      "มันรู้จักสัญลักษณ์แค่ $1 ตัว เลือกมาหนึ่งตัว หรือระบายและลบพิกเซลบนตาราง แล้วดูสัญญาณไหลผ่านโครงข่าย"],
    ["^(\\d+) pixels in$", "อินพุต $1 พิกเซล"],
    ["^(\\d+) neurons$", "นิวรอน $1 ตัว"],
    ["^(\\d+) answers$", "$1 คำตอบ"],
    ["^First-layer neuron (\\d+): show the pixels it looks for$", "นิวรอนชั้นแรกตัวที่ $1: แสดงพิกเซลที่มันมองหา"],
    ["^It thinks: (.+) \\((.+)\\), (\\S+) sure$", "มันคิดว่าเป็น $1 ($2) มั่นใจ $3"],
    ["^A small neural network\\. The grid is empty: paint something\\.$", "โครงข่ายประสาทเทียมขนาดเล็ก ตารางว่างเปล่า ลองระบายอะไรสักอย่างดู"],
    ["^A small neural network\\. It thinks: (.+) \\((.+)\\), (\\S+) sure$", "โครงข่ายประสาทเทียมขนาดเล็ก มันคิดว่าเป็น $1 ($2) มั่นใจ $3"],
    ["^A small neural network\\. (.+)$", "โครงข่ายประสาทเทียมขนาดเล็ก $1"],
    ["^Neuron (\\d+) looks for: blue = ink wanted, purple = ink unwanted$", "นิวรอน $1 มองหา: น้ำเงิน = ต้องมีหมึก, ม่วง = ต้องไม่มีหมึก"],

    // nn.js: inside the real reader
    ["^What the reader computes for (.+), stage by stage\\. Move along the figure to pick a step\\.$",
      "สิ่งที่ตัวอ่านคำนวณสำหรับ $1 ทีละชุดชั้น เลื่อนไปตามรูปเพื่อเลือกก้าว"],
    ["^the strip · (\\d+) × (\\d+) pixels$", "แถบภาพ · $1 × $2 พิกเซล"],
    ["^Stage (\\d+)$", "ชุดชั้นที่ $1"],
    ["^(\\d+) feature maps · (\\d+) × (\\d+) · 3 shown$", "แผนที่ฟีเจอร์ $1 แผ่น · $2 × $3 · แสดง 3 แผ่น"],
    ["^(\\d+) numbers for each of the (\\d+) steps$", "ตัวเลข $1 ตัวสำหรับแต่ละก้าวจาก $2 ก้าว"],
    ["^(\\d+) possible answers · the (\\d+) strongest shown$", "คำตอบที่เป็นไปได้ $1 แบบ · แสดง $2 อันดับที่แรงที่สุด"],
    ["^Step (\\d+) of (\\d+) · it pays most attention to pixels (\\d+)–(\\d+) of the strip · best answer: nothing here \\((.+)\\)$",
      "ก้าวที่ $1 จาก $2 · สนใจพิกเซลที่ $3–$4 ของแถบมากที่สุด · คำตอบที่ดีที่สุด: ไม่มีอะไร ($5)"],
    ["^Step (\\d+) of (\\d+) · it pays most attention to pixels (\\d+)–(\\d+) of the strip · best answer: space \\((.+)\\)$",
      "ก้าวที่ $1 จาก $2 · สนใจพิกเซลที่ $3–$4 ของแถบมากที่สุด · คำตอบที่ดีที่สุด: เว้นวรรค ($5)"],
    ["^Step (\\d+) of (\\d+) · it pays most attention to pixels (\\d+)–(\\d+) of the strip · best answer: (.+) \\((.+)\\)$",
      "ก้าวที่ $1 จาก $2 · สนใจพิกเซลที่ $3–$4 ของแถบมากที่สุด · คำตอบที่ดีที่สุด: $5 ($6)"],

    // deep.js: marks, architecture labels
    ["^chart: (.+)$", "ตามตาราง: $1"],
    ["^ぱ: ring r (.+)$", "ぱ: วงกลม r $1"],
    ["^line (\\S+) × stroke$", "เส้นหนา $1 × เส้นกลิฟ"],
    ["^stage (\\d)$", "สเตจ $1"],
    ["^stage (\\d) · (s\\d+)$", "สเตจ $1 · $2"],
    ["^up (×\\d)$", "ขยาย $1"],
    ["^gray · (.+)$", "ภาพเทา · $1"],
    ["^2 convs (s\\d,\\d) · (.+)$", "conv 2 ชั้น $1 · $2"],
    ["^6 UIB (s\\d,\\d) · (.+)$", "UIB 6 บล็อก $1 · $2"],
    ["^avg pool · (.+)$", "พูลเฉลี่ย · $1"],
    ["^2 × res\\. (.+)$", "res. 2 บล็อก $1"],
    ["^D = (\\S+) h · kernel (\\S+) h thick · unclip 1\\.5 → (\\S+) h · exact inverse → 1\\.000 h$",
      "D = $1 h · เคอร์เนลหนา $2 h · unclip 1.5 → $3 h · ตัวผกผันแม่นตรง → 1.000 h"],

    // deep.js: int8 grids (options, row labels, readout)
    ["^input \\(fixed\\) · e = (-?\\d+)$", "อินพุต (ตรึงค่า) · e = $1"],
    ["^stem · e = (-?\\d+)$", "สเต็ม · e = $1"],
    ["^mean over H · e = (-?\\d+)$", "เฉลี่ยตาม H · e = $1"],
    ["^classifier · e = (-?\\d+)$", "ตัวจำแนก · e = $1"],
    ["^add (\\d+) · e = (-?\\d+)$", "add $1 · e = $2"],
    ["^int8 grids around exponent (-?\\d+)$", "กริด int8 รอบเลขชี้กำลัง $1"],
    ["^step (\\S+) · max (\\S+) · every 2nd tick$", "ขั้น $1 · สูงสุด $2 · แสดงทุก 2 ขีด"],
    ["^step (\\S+) · max (\\S+) · every (\\d+)th tick$", "ขั้น $1 · สูงสุด $2 · แสดงทุก $3 ขีด"],
    ["^step (\\S+) · max (\\S+)$", "ขั้น $1 · สูงสุด $2"],
    ["^input \\(fixed\\) · e = (-?\\d+) · step 2\\^(-?\\d+) = (\\S+) · int8 range (\\[[^\\]]+\\]) · input exponent fixed at −7$",
      "อินพุต (ตรึงค่า) · e = $1 · ขั้น 2^$2 = $3 · ช่วง int8 $4 · เลขชี้กำลังอินพุตตรึงไว้ที่ −7"],
    ["^stem" + GRID_TAIL, "สเต็ม" + GRID_OUT],
    ["^mean over H" + GRID_TAIL, "เฉลี่ยตาม H" + GRID_OUT],
    ["^classifier" + GRID_TAIL, "ตัวจำแนก" + GRID_OUT],
    ["^(\\S+(?: \\S+)?)" + GRID_TAIL, "$1 · e = $2 · ขั้น 2^$3 = $4 · ช่วง int8 $5 · ค่าตัด KL c = $6 · log2(c/128) = $7 → $8"]
  ];

  // deep.js: exponent readout, one pattern per kind, with and without the left-shift note
  KINDS.forEach(function (k) {
    patterns.push(["^(\\S+) · " + k[0] + EXP_TAIL + " \\(left shift\\)$", "$1 · " + k[1] + EXP_OUT + " (เลื่อนซ้าย)"]);
    patterns.push(["^(\\S+) · " + k[0] + EXP_TAIL + "$", "$1 · " + k[1] + EXP_OUT]);
  });

  patterns.push(
    // deep.js: QAT and round curves
    ["^QAT fine-tune: 48 px horizontal \\((.+)\\)$", "การปรับละเอียด QAT: 48 px แนวนอน ($1)"],
    ["^QAT fine-tune: 48 px vertical \\((.+)\\)$", "การปรับละเอียด QAT: 48 px แนวตั้ง ($1)"],
    ["^QAT fine-tune: 32 px three-stage \\((.+)\\)$", "การปรับละเอียด QAT: 32 px สามสเตจ ($1)"],
    ["^QAT fine-tune: (.+)$", "การปรับละเอียด QAT: $1"],
    ["^Round (\\d)$", "รอบที่ $1"],
    ["^step (\\S+) · int8 (\\S+) · float (\\S+)$", "สเต็ป $1 · int8 $2 · float $3"],
    ["^step (\\S+) · mnv4s, ImageNet-pretrained (\\S+) · mnv4s, from scratch (\\S+) · tiny CNN, from scratch (\\S+)$",
      "สเต็ป $1 · mnv4s, ฝึกล่วงหน้าด้วย ImageNet $2 · mnv4s, ฝึกจากศูนย์ $3 · CNN จิ๋ว, ฝึกจากศูนย์ $4"],
    ["^step (\\S+) · mnv4s, 48 px (\\S+) · mnv4s, 32 px (\\S+) · 3-stage mnv4s, 32 px (\\S+)$",
      "สเต็ป $1 · mnv4s, 48 px $2 · mnv4s, 32 px $3 · mnv4s 3 สเตจ, 32 px $4"],
    ["^step (\\S+) · mnv4s, 48 px, horizontal (\\S+) · mnv4s, 48 px, vertical columns (\\S+) · 3-stage mnv4s, 32 px, horizontal (\\S+)$",
      "สเต็ป $1 · mnv4s, 48 px, แนวนอน $2 · mnv4s, 48 px, คอลัมน์แนวตั้ง $3 · mnv4s 3 สเตจ, 32 px, แนวนอน $4"],
    ["^step (\\S+) · F1 (\\S+) · precision (\\S+) · recall (\\S+)$", "สเต็ป $1 · F1 $2 · ความแม่นยำ $3 · ความครบถ้วน $4"],
    ["^step (\\S+) · (.+)$", "สเต็ป $1 · $2"],

    // deep.js: confusions
    ["^(\\S+) read as (\\S+), (\\d+) times$", "$1 ถูกอ่านเป็น $2 จำนวน $3 ครั้ง"],
    ["^(\\d+) times$", "$1 ครั้ง"],

    // deep.js: confidence filter readouts and table
    ["^min confidence none · precision (\\S+) · recall (\\S+) · F1 (\\S+)$", "ความมั่นใจขั้นต่ำ ไม่มี · ความแม่นยำ $1 · ความครบถ้วน $2 · F1 $3"],
    ["^min confidence (\\S+) · precision (\\S+) · recall (\\S+) · F1 (\\S+)$", "ความมั่นใจขั้นต่ำ $1 · ความแม่นยำ $2 · ความครบถ้วน $3 · F1 $4"],
    ["^min confidence none · CER h (\\S+) · CER v (\\S+) · matched (\\S+)$", "ความมั่นใจขั้นต่ำ ไม่มี · CER h $1 · CER v $2 · จับคู่ได้ $3"],
    ["^min confidence (\\S+) · CER h (\\S+) · CER v (\\S+) · matched (\\S+)$", "ความมั่นใจขั้นต่ำ $1 · CER h $2 · CER v $3 · จับคู่ได้ $4"],
    ["^(\\S+) \\(demo\\)$", "$1 (เดโม)"],

    // deep.js: page timings
    ["^read (\\d+) lines$", "อ่าน $1 บรรทัด"],
    ["^letterbox (\\S+) ms, detector \\(int8\\) (\\S+) ms, post-processing (\\S+) ms, read (\\d+) lines (\\S+) ms$",
      "เลตเตอร์บ็อกซ์ $1 ms, ตัวตรวจจับ (int8) $2 ms, ประมวลผลภายหลัง $3 ms, อ่าน $4 บรรทัด $5 ms"],
    ["^(\\d+) × (\\d+) px page, (\\d+) detected lines; node, one thread, wasm SIMD, CPU shared; median of (\\d+) runs$",
      "หน้า $1 × $2 px, ตรวจพบ $3 บรรทัด; node, หนึ่งเธรด, wasm SIMD, ใช้ CPU ร่วมกับงานอื่น; มัธยฐานจาก $4 ครั้ง"],
    ["^Total (\\S+) ms; one line \\(crop, prepare, read\\) (\\S+) ms on its own\\. Source: img/pipeline/pipeline\\.json\\.$",
      "รวม $1 ms; หนึ่งบรรทัด (ตัด เตรียม อ่าน) ใช้ $2 ms เมื่อรันเดี่ยว ๆ ที่มา: img/pipeline/pipeline.json"]
  );

  window.HXH_I18N = { strings: strings, patterns: patterns };
})();
