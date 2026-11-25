// Japanese translations of the text web/app.js draws into the reader page (index-ja.html).
// web/i18n.js applies them: an exact match on the trimmed text of a text node or of an
// aria-label / alt / title / placeholder, else the first matching pattern.
// Readings (kana, romaji) and numbers are never translated: every pattern is anchored on English
// words that a reading cannot contain.
window.HXH_I18N = {
  strings: {
    // ------------------------------------------------------------ MSG (toasts)
    "This browser could not open that image. Try a JPEG or PNG.": "このブラウザではその画像を開けませんでした。JPEG か PNG で試してください。",
    "That file isn't an image. Choose, paste or drop a JPEG, PNG or WebP picture.": "そのファイルは画像ではありません。JPEG、PNG、WebP の画像を選ぶか、貼り付けるか、ドロップしてください。",
    "There's no image on the clipboard. Copy an image first, then try again.": "クリップボードに画像がありません。先に画像をコピーしてから、もう一度試してください。",
    "The browser didn't let this page read the clipboard. Paste with the keyboard (Ctrl+V or ⌘V), or choose the image instead.": "ブラウザがこのページにクリップボードの読み取りを許可しませんでした。キーボードで貼り付ける（Ctrl+V または ⌘V）か、画像を選んでください。",
    "That link isn't an image. Drag the picture itself, or save it and choose the file.": "そのリンクは画像ではありません。画像そのものをドラッグするか、保存してからファイルを選んでください。",
    "The site that image comes from doesn't let other pages read it. Save the image to your device, then drop or choose the file.": "その画像のサイトは、ほかのページからの読み込みを許可していません。画像を端末に保存してから、ファイルをドロップするか選んでください。",
    "Couldn't copy. Select the text and copy it yourself.": "コピーできませんでした。文字を選択して、手動でコピーしてください。",

    // ------------------------------------------------------------ loading, status, model info
    "All int8, WebAssembly SIMD.": "すべて int8、WebAssembly SIMD。",
    "All int8, WebAssembly (no SIMD).": "すべて int8、WebAssembly（SIMD なし）。",
    "The accurate models could not be loaded, so Fast is used.": "高精度モデルを読み込めなかったため、高速モデルを使っています。",
    "Ready. Everything runs on this device.": "準備ができました。すべてこの端末の中で動きます。",
    "Live mode needs the line detector, which did not load.": "ライブモードには行さがしAIが必要ですが、読み込めませんでした。",
    "Loading the models…": "モデルを読み込んでいます…",
    "Loading the reader models…": "読み取りモデルを読み込んでいます…",

    // ------------------------------------------------------------ live camera
    "The camera isn't allowed here (blocked by the browser or by the page that hosts this demo).": "ここではカメラが許可されていません（ブラウザか、このデモを載せているページがブロックしています）。",
    "No camera was found.": "カメラが見つかりませんでした。",
    "This browser can't open a camera on this page (it needs a secure https:// address).": "このブラウザでは、このページからカメラを開けません（安全な https:// のアドレスが必要です）。",
    "Live mode needs a camera.": "ライブモードにはカメラが必要です。",
    "Showing live mode on a moving example page instead, as if it were the camera.": "代わりに、動くサンプルページをカメラの映像に見立ててライブモードを表示しています。",
    "Point the camera at Hunter × Hunter script. Tap a line (or Freeze) to keep the picture.": "カメラをハンター文字に向けてください。行をタップする（または「静止画にする」を押す）と、その画面が残ります。",
    "Paused while the page is hidden.": "ページが表示されていない間は一時停止しています。",
    "Starting…": "起動中…",

    // ------------------------------------------------------------ reading, hints, summary
    "Finding the lines…": "行をさがしています…",
    "Looking for a line here…": "ここに行がないか探しています…",
    "No line found at that spot. Drag a box around the text to read it anyway.": "その場所には行が見つかりませんでした。文字を枠で囲めば、そのまま読み取れます。",
    "Tap a numbered line to see it. Tap unmarked text to look closer, or drag a box around one line to read just that.": "番号の付いた行をタップすると表示されます。枠のない文字をタップすると近くでもう一度探し、1 行を枠で囲むとその行だけを読みます。",
    "Drag a box around one line of text to read it.": "1 行の文字を枠で囲むと読み取ります。",
    "Drag a box around one line of text.": "1 行の文字を枠で囲んでください。",
    "Show all": "全体を表示",
    "Zoom to line": "行を拡大",

    // ------------------------------------------------------------ the list of lines
    "(nothing recognised)": "（読み取れませんでした）",
    "not Hunter script?": "ハンター文字ではない？",
    "vertical": "縦書き",
    "tapped": "タップで検出",
    "box": "枠",
    "not Hunter script? · vertical": "ハンター文字ではない？ · 縦書き",
    "vertical · tapped": "縦書き · タップで検出",
    "vertical · box": "縦書き · 枠",

    // ------------------------------------------------------------ the selected line
    "No text recognised in this line.": "この行からは文字を読み取れませんでした。",
    ", runner-up": "、次の候補",
    "Label:": "正解:",
    "✓ match": "✓ 一致",
    "✗ differs": "✗ 不一致",
    "What the reader saw.": "読み取りAIが見たもの。",
    "The text is small in this image, so it was scaled up. Get closer (or use a sharper image) so the small dakuten marks survive.": "この画像では文字が小さいので、拡大して読みました。小さな濁点がつぶれないよう、もっと近づくか、もっと鮮明な画像を使ってください。",
    "No vertical reader is loaded, so this column was read by the horizontal one.": "縦書き用の読み取りAIが読み込まれていないため、この列は横書き用の読み取りAIで読みました。",
    "Auto-fit: read the box tightened around the text.": "自動調整: 文字に合わせて縮めた枠を読みました。",
    "Found by looking closer around your tap.": "タップした場所の周りを詳しく見て見つけた行です。",

    // ------------------------------------------------------------ the site's language menu (web/lang.js)
    "Translation coming soon": "翻訳は準備中です",
  },

  patterns: [
    // status and model info
    ["^Could not load the model: (.*)\\. Open this page through a web server, or use the single-file build\\.$",
      "モデルを読み込めませんでした: $1。Web サーバー経由でこのページを開くか、単一ファイル版を使ってください。"],
    ["^Could not load the model: (.*)\\.$", "モデルを読み込めませんでした: $1。"],
    ["^Reading failed: (.*)$", "読み取りに失敗しました: $1"],
    ["^Fast models: line detector (\\S+) \\(([\\d,]+) KB, ([\\d.]+) G MACs at (\\d+)×(\\d+)\\), horizontal reader (\\S+) \\(([\\d,]+) KB\\), vertical reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "高速モデル: 行さがしAI $1（$2 KB、$4×$5 で $3 G MACs）、横書き用読み取りAI $6（$7 KB）、縦書き用読み取りAI $8（$9 KB）。読み取りAIは $11×$12 で $10 M MACs。"],
    ["^Fast models: horizontal reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "高速モデル: 横書き用読み取りAI $1（$2 KB）。読み取りAIは $4×$5 で $3 M MACs。"],
    ["^Accurate models: line detector (\\S+) \\(([\\d,]+) KB, ([\\d.]+) G MACs at (\\d+)×(\\d+)\\), horizontal reader (\\S+) \\(([\\d,]+) KB\\), vertical reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "高精度モデル: 行さがしAI $1（$2 KB、$4×$5 で $3 G MACs）、横書き用読み取りAI $6（$7 KB）、縦書き用読み取りAI $8（$9 KB）。読み取りAIは $11×$12 で $10 M MACs。"],
    ["^Accurate models: horizontal reader (\\S+) \\(([\\d,]+) KB\\); readers (\\d+) M MACs at (\\d+)×(\\d+)\\.$",
      "高精度モデル: 横書き用読み取りAI $1（$2 KB）。読み取りAIは $4×$5 で $3 M MACs。"],

    // examples
    ["^Example page (\\d+)$", "サンプルページ $1"],
    ["^Example line (\\d+)$", "サンプルの行 $1"],

    // live camera: the reason there is no camera, then what happens instead
    ["^The camera isn't allowed here \\(blocked by the browser or by the page that hosts this demo\\)\\. Showing live mode on a moving example page instead, as if it were the camera\\.$",
      "ここではカメラが許可されていません（ブラウザか、このデモを載せているページがブロックしています）。代わりに、動くサンプルページをカメラの映像に見立ててライブモードを表示しています。"],
    ["^No camera was found\\. Showing live mode on a moving example page instead, as if it were the camera\\.$",
      "カメラが見つかりませんでした。代わりに、動くサンプルページをカメラの映像に見立ててライブモードを表示しています。"],
    ["^This browser can't open a camera on this page \\(it needs a secure https:// address\\)\\. Showing live mode on a moving example page instead, as if it were the camera\\.$",
      "このブラウザでは、このページからカメラを開けません（安全な https:// のアドレスが必要です）。代わりに、動くサンプルページをカメラの映像に見立ててライブモードを表示しています。"],
    ["^The camera could not be opened \\((\\w*)\\)\\. Showing live mode on a moving example page instead, as if it were the camera\\.$",
      "カメラを開けませんでした（$1）。代わりに、動くサンプルページをカメラの映像に見立ててライブモードを表示しています。"],
    ["^The camera isn't allowed here \\(blocked by the browser or by the page that hosts this demo\\)\\. Live mode needs a camera\\.$",
      "ここではカメラが許可されていません（ブラウザか、このデモを載せているページがブロックしています）。ライブモードにはカメラが必要です。"],
    ["^No camera was found\\. Live mode needs a camera\\.$", "カメラが見つかりませんでした。ライブモードにはカメラが必要です。"],
    ["^This browser can't open a camera on this page \\(it needs a secure https:// address\\)\\. Live mode needs a camera\\.$",
      "このブラウザでは、このページからカメラを開けません（安全な https:// のアドレスが必要です）。ライブモードにはカメラが必要です。"],
    ["^The camera could not be opened \\((\\w*)\\)\\. Live mode needs a camera\\.$", "カメラを開けませんでした（$1）。ライブモードにはカメラが必要です。"],
    ["^The camera could not be opened \\((\\w*)\\)\\.$", "カメラを開けませんでした（$1）。"],
    ["^([\\d.]+) frames/s · detect (\\d+) ms · read (\\d+) ms \\((\\d+) kept\\)$", "毎秒 $1 フレーム · 検出 $2 ms · 読み取り $3 ms（$4 行は前の読みを使用）"],
    ["^([\\d.]+) frames/s · detect (\\d+) ms · read (\\d+) ms$", "毎秒 $1 フレーム · 検出 $2 ms · 読み取り $3 ms"],

    // the page summary
    ["^(\\d+) lines? found in (\\d+) ms, read in (\\d+) ms \\(([\\d.]+) ms per line\\)\\.$", "$1 行を $2 ms で見つけ、$3 ms で読みました（1 行あたり $4 ms）。"],
    ["^No Hunter script found \\((\\d+) ms\\)\\. Tap the text to look closer, or drag a box around one line\\.$",
      "ハンター文字は見つかりませんでした（$1 ms）。文字をタップすると近くでもう一度探します。1 行を枠で囲んで読むこともできます。"],
    ["^No Hunter script found \\((\\d+) ms\\)\\.$", "ハンター文字は見つかりませんでした（$1 ms）。"],

    // the selected line: unsure characters (aria-label, tooltip, the detail line)
    ["^Unsure: (\\S+) (\\d+%), maybe (\\S+) \\((\\d+%)\\)$", "自信なし: $1 $2、もしかすると $3（$4）"],
    ["^Unsure: (\\S+) (\\d+%)$", "自信なし: $1 $2"],
    ["^(\\S+) (\\d+%), maybe (\\S+) \\((\\d+%)\\)$", "$1 $2、もしかすると $3（$4）"],
    ["^(\\d+)% sure$", "自信 $1%"],

    // the strip the reader saw
    ["^What the vertical reader saw: the column turned on its side, as a (\\d+) × (\\d+) px grayscale strip\\.$",
      "縦書き用の読み取りAIが見たもの: 列を横に倒し、$1 × $2 px のグレースケールの帯にしたもの。"],
    ["^What the horizontal reader saw: the column turned on its side, as a (\\d+) × (\\d+) px grayscale strip\\.$",
      "横書き用の読み取りAIが見たもの: 列を横に倒し、$1 × $2 px のグレースケールの帯にしたもの。"],
    ["^What the horizontal reader saw: the line as a (\\d+) × (\\d+) px grayscale strip\\.$",
      "横書き用の読み取りAIが見たもの: 行を $1 × $2 px のグレースケールの帯にしたもの。"],

    // notices
    ["^This line is (\\d+) px long at (\\d+) px height, more than the reader's (\\d+) px, so it was squeezed ([\\d.]+)× to fit\\. Expect errors: drag boxes around about (\\d+) glyphs at a time\\.$",
      "この行は高さ $2 px で長さが $1 px あり、読み取りAIの $3 px を超えるため、$4 分の 1 に縮めて収めました。まちがいが出やすいので、一度に $5 文字くらいずつ枠で囲んでください。"],
    ["^The reader is unsure of this line \\((\\d+)% on average\\): it may not be Hunter script at all \\(Latin or Japanese text, art\\)\\.$",
      "読み取りAIはこの行に自信がありません（平均 $1%）。ハンター文字ではないかもしれません（ラテン文字や日本語の文字、絵など）。"],

    // the line counter under the result
    ["^Line (\\d+) of (\\d+) · WebAssembly SIMD$", "$1 行目 / 全 $2 行 · WebAssembly SIMD"],
    ["^Line (\\d+) of (\\d+) · WebAssembly$", "$1 行目 / 全 $2 行 · WebAssembly"],
  ],
};
