//! Foreign (non-Hunter) text for the distractors: English-like prose, Hepburn romaji (with
//! macrons), URLs, numbers / dates / prices, Japanese (kana, kanji sentences, full-width
//! punctuation), a little Korean, Greek and Cyrillic, plus SFX and labels. None of it is
//! labelled: it only has to look like real writing in real fonts. See `FOREIGN.md`.

use crate::fonts;
use crate::rng::Rng;
use crate::text::TextSampler;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Script {
    Latin,
    Romaji,
    Url,
    Digits,
    Japanese,
    Kana,
    Korean,
    Greek,
    Cyrillic,
    Icon,
}

impl Script {
    pub fn name(self) -> &'static str {
        match self {
            Script::Latin => "latin",
            Script::Romaji => "romaji",
            Script::Url => "url",
            Script::Digits => "digits",
            Script::Japanese => "japanese",
            Script::Kana => "kana",
            Script::Korean => "korean",
            Script::Greek => "greek",
            Script::Cyrillic => "cyrillic",
            Script::Icon => "icon",
        }
    }

    /// Font coverage bits the script needs.
    pub fn need(self) -> u8 {
        match self {
            Script::Latin | Script::Url | Script::Digits => fonts::LATIN,
            Script::Romaji => fonts::LATIN | fonts::LATIN_EXT,
            Script::Japanese | Script::Kana => fonts::JAPANESE,
            Script::Korean => fonts::KOREAN,
            Script::Greek => fonts::GREEK,
            Script::Cyrillic => fonts::CYRILLIC,
            Script::Icon => 0,
        }
    }

    /// Wrapped / cut anywhere (no spaces between words).
    pub fn by_char(self) -> bool {
        matches!(self, Script::Japanese | Script::Kana)
    }
}

/// Order used by the per-kind script weight tables.
pub const SCRIPTS: [Script; 9] =
    [Script::Latin, Script::Romaji, Script::Url, Script::Digits, Script::Japanese, Script::Kana, Script::Korean, Script::Greek, Script::Cyrillic];

/// What the text is for: decides its length and shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Sentence,
    Phrase,
    Heading,
    Sfx,
    Label,
    Word,
    Link,
    Paragraph,
}

const EN: &str = "the of and to in is was for on that with as by it at from his her an be this are which or had not but were they their \
one all also have has more first after other its new been into time two would can only who when there may over most some such these than \
then them about between during under while where years year used being made many three later both known part each became since against well \
because before through same any own however series name called world people work number found several including another among early long form \
game anime manga story chapter volume episode character characters alphabet letter letters script writing system hunter hunters license exam \
association language japanese english hiragana katakana kanji syllable syllables sound sounds vowel consonant table chart sample text page wiki \
edit history source read more see also external links references notes contents main article example examples symbol symbols glyph glyphs \
shape shapes based simple cipher code fictional used appears appeared shows written write writes reading translation translated meaning means \
word words sentence original official design designed creator author artist published publisher magazine weekly adaptation television film fans \
community site website online free download image images file files user users comment comments reply share like follow home search menu login \
account help contact about privacy terms copyright rights reserved update updated news latest posts post category tags archive month day today \
week morning night city island tower ship train station road school house room door window light dark black white red blue green small large \
big little old young good great best last next previous back top bottom left right open close start end please thank you welcome hello yes no \
what why how who where when every each mark marks dot dots small kana long vowel table below above column row";

const NAMES: &[&str] = &[
    "Gon", "Killua", "Kurapika", "Leorio", "Hisoka", "Netero", "Ging", "Greed Island", "Yorknew", "Zoldyck", "Togashi", "Shueisha", "Madhouse",
    "Omniglot", "Hunterpedia", "Weekly Shonen Jump", "Hunter Association", "Phantom Troupe", "Chimera Ant", "Heaven's Arena", "Whale Island", "Nen",
];

const SFX_LATIN: &[&str] = &[
    "BAM!", "WHAM!", "BOOM!", "KRAK!", "THUD", "SWISH", "WHOOSH", "SLAM!", "CRASH!", "GASP!", "HUH?!", "WHAT?!", "HA HA HA", "HEH", "ZZZ", "SNAP",
    "CLICK", "BOING", "POW!", "ZOOM", "RUMBLE", "SHFF", "TAP TAP", "BA-DUMP", "GRRR", "ARGH!", "OOF", "!!", "?!", "...!", "WOW!", "NO WAY!",
];
const SFX_ROMAJI: &[&str] =
    &["DOKI DOKI", "ZAWA ZAWA", "GOGOGO", "DON!", "BAN!", "GOGOGOGO", "SHIIN", "PIKU", "GORO GORO", "DOKAN!", "ZUBAAN", "KIRA KIRA", "NIKO", "BACHI", "GASHI"];
const SFX_KANA: &[&str] = &[
    "ドン", "ゴゴゴ", "バン", "ドキドキ", "ザワザワ", "シーン", "ゴロゴロ", "ドカーン", "ズバッ", "キラキラ", "ニコッ", "バチッ", "ガシッ", "ゴゴゴゴ", "ピクッ", "ドドド",
    "ズズズ", "ハッ", "えっ", "ええっ!?", "なにっ!?", "ふふふ", "あはは", "うわあ", "ひっ",
];

const JA_N: &[&str] = &[
    "文字", "日本語", "漫画", "作品", "作者", "物語", "世界", "主人公", "試験", "協会", "言葉", "表", "仮名", "平仮名", "片仮名", "漢字", "発音", "意味", "名前", "記号",
    "暗号", "文章", "単語", "母音", "子音", "濁点", "半濁点", "長音", "例", "資料", "説明", "一覧", "画像", "書体", "公式", "連載", "雑誌", "週刊", "少年", "番組",
    "放送", "原作", "映画", "声優", "登場人物", "念能力", "会長", "試験官", "受験者", "島", "旅団", "友達", "父親", "家族", "時間", "場所", "情報", "内容", "方法",
    "目的", "結果", "問題", "答え", "今日", "明日", "東京", "大阪", "京都", "学校", "先生", "学生", "会社", "電車", "駅", "天気", "春", "夏", "秋", "冬", "山", "川",
    "海", "空", "花", "猫", "犬", "本", "新聞", "写真", "音楽", "料理", "店", "道", "車", "手紙", "部屋", "窓", "読者", "解説", "対応表", "五十音", "書き方",
];
const JA_K: &[&str] = &[
    "ハンター", "アルファベット", "ライセンス", "カタカナ", "ゲーム", "アニメ", "キャラクター", "フォント", "データ", "テキスト", "ページ", "リンク", "サイト", "ニュース",
    "メニュー", "ホーム", "ログイン", "ダウンロード", "コメント", "シリーズ", "ストーリー", "ファン", "イラスト", "グリードアイランド", "キメラアント", "ゴン", "キルア",
    "クラピカ", "レオリオ", "ヒソカ", "ネテロ", "ジン", "ヨークシン", "ハンター試験", "ハンター文字",
];
const JA_P: &[&str] = &["は", "が", "を", "に", "の", "で", "と", "も", "へ", "から", "まで", "より"];
const JA_V: &[&str] = &[
    "書く", "読む", "使う", "見る", "作る", "知る", "呼ぶ", "表す", "書きます", "読みます", "使われています", "知られている", "呼ばれる", "登場する", "表記される",
    "対応している", "存在する", "掲載された", "発表された", "放送された", "描かれている", "使用される",
];
const JA_E: &[&str] = &["です。", "でした。", "である。", "ではない。", "だ。", "とされる。", "のようだ。", "らしい。", "について。", "がある。", "になった。"];
const JA_HEAD: &[&str] =
    &["概要", "歴史", "文字一覧", "使い方", "関連項目", "脚注", "外部リンク", "目次", "参考文献", "ハンター文字", "ハンター×ハンター", "五十音表", "読み方", "書き方"];
const JA_LABEL: &[&str] = &["出口", "入口", "営業中", "立入禁止", "非常口", "東口", "西口", "受付", "準備中", "本日休業", "お知らせ", "注意", "危険", "禁煙"];

const KO: &[&str] = &[
    "한국어", "문자", "사냥꾼", "알파벳", "만화", "애니메이션", "일본어", "히라가나", "가타카나", "이름", "세계", "이야기", "작가", "표", "글자", "발음", "예시", "페이지",
    "링크", "검색", "로그인", "사용자", "정보", "그리고", "하지만", "있습니다", "합니다", "입니다", "사람", "시간", "오늘", "내일", "학교", "친구", "우리", "이것",
    "무엇", "어디", "언제", "좋아요", "감사합니다", "안녕하세요", "시험", "협회", "섬", "게임", "소년", "주간", "연재",
];
const EL: &[&str] = &[
    "αλφάβητο", "γράμμα", "γράμματα", "κυνηγός", "ιαπωνικά", "κείμενο", "σελίδα", "παράδειγμα", "και", "το", "η", "ο", "είναι", "για", "με", "στο", "από",
    "γλώσσα", "ιστορία", "κόσμος", "όνομα", "χρόνος", "ημέρα", "νέα", "σύστημα", "γραφής", "χαρακτήρες", "σειρά", "επεισόδιο", "πίνακας",
];
const RU: &[&str] = &[
    "алфавит", "буква", "буквы", "охотник", "японский", "текст", "страница", "пример", "и", "в", "на", "это", "для", "с", "из", "язык", "история", "мир",
    "имя", "время", "день", "новости", "письмо", "слово", "читать", "писать", "манга", "аниме", "система", "письменности", "символы", "серия", "глава", "том",
];

const DOMAINS: &[&str] = &[
    "omniglot", "hunterxhunter", "hunterpedia", "fandom", "wikipedia", "example", "manga-news", "anime", "jump", "togashi", "alphabet", "fonts", "kana",
    "nihongo", "hxh", "news", "blog", "forum", "archive", "scripts", "writing-systems",
];
const TLDS: &[&str] = &[".com", ".org", ".net", ".jp", ".co.jp", ".io", ".wiki", ".info", ".fandom.com"];
const PATHS: &[&str] = &[
    "/wiki/", "/writing/", "/fonts/", "/news/", "/2011/10/", "/en/", "/ja/", "/index.html", "?page=2", "#history", "/Hunter_x_Hunter_alphabet",
    "hunterxhunter.htm", "/chapter-340", "/images/chart.png", "/tag/hunter", "/search?q=hunter+alphabet", "/forum/t/12345", "/about",
];
const LINKS: &[&str] = &[
    "Read more »", "See also", "Edit", "Next page →", "← Previous", "Share", "Reply", "Log in", "Sign up", "Download", "Contact us", "Privacy policy",
    "Terms of use", "Back to top ↑", "Show all", "Load more", "View source", "History", "Talk", "Random page", "Help", "Subscribe", "Home", "Sitemap",
];
const MONTHS: &[&str] = &["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

fn en_words() -> Vec<&'static str> {
    EN.split_whitespace().collect()
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn title(s: &str) -> String {
    s.split(' ').map(|w| if w.len() > 3 || w.chars().next().is_some_and(|c| c.is_uppercase()) { cap(w) } else { w.to_string() }).collect::<Vec<_>>().join(" ")
}

pub(crate) fn kata(s: &str) -> String {
    s.chars().map(|c| if ('\u{3041}'..='\u{3096}').contains(&c) { char::from_u32(c as u32 + 0x60).unwrap_or(c) } else { c }).collect()
}

/// Hiragana -> Hepburn romaji: yōon, sokuon, n' before vowels, macrons for ー (and often for
/// ou / uu).
pub fn romaji(kana: &str) -> String {
    const T: &[(char, &str)] = &[
        ('あ', "a"), ('い', "i"), ('う', "u"), ('え', "e"), ('お', "o"), ('か', "ka"), ('き', "ki"), ('く', "ku"), ('け', "ke"), ('こ', "ko"),
        ('さ', "sa"), ('し', "shi"), ('す', "su"), ('せ', "se"), ('そ', "so"), ('た', "ta"), ('ち', "chi"), ('つ', "tsu"), ('て', "te"), ('と', "to"),
        ('な', "na"), ('に', "ni"), ('ぬ', "nu"), ('ね', "ne"), ('の', "no"), ('は', "ha"), ('ひ', "hi"), ('ふ', "fu"), ('へ', "he"), ('ほ', "ho"),
        ('ま', "ma"), ('み', "mi"), ('む', "mu"), ('め', "me"), ('も', "mo"), ('や', "ya"), ('ゆ', "yu"), ('よ', "yo"), ('ら', "ra"), ('り', "ri"),
        ('る', "ru"), ('れ', "re"), ('ろ', "ro"), ('わ', "wa"), ('を', "o"), ('ん', "n"), ('が', "ga"), ('ぎ', "gi"), ('ぐ', "gu"), ('げ', "ge"),
        ('ご', "go"), ('ざ', "za"), ('じ', "ji"), ('ず', "zu"), ('ぜ', "ze"), ('ぞ', "zo"), ('だ', "da"), ('ぢ', "ji"), ('づ', "zu"), ('で', "de"),
        ('ど', "do"), ('ば', "ba"), ('び', "bi"), ('ぶ', "bu"), ('べ', "be"), ('ぼ', "bo"), ('ぱ', "pa"), ('ぴ', "pi"), ('ぷ', "pu"), ('ぺ', "pe"),
        ('ぽ', "po"), ('ゔ', "vu"), ('ゐ', "i"), ('ゑ', "e"), ('ぁ', "a"), ('ぃ', "i"), ('ぅ', "u"), ('ぇ', "e"), ('ぉ', "o"), ('ゎ', "wa"),
    ];
    let macron = |c: char| match c {
        'a' => 'ā',
        'i' => 'ī',
        'u' => 'ū',
        'e' => 'ē',
        'o' => 'ō',
        x => x,
    };
    let mut out = String::new();
    let mut sokuon = false;
    let chars: Vec<char> = kana.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            ' ' => out.push(' '),
            'っ' => sokuon = true,
            'ー' => {
                if let Some(l) = out.pop() {
                    out.push(macron(l));
                }
            }
            'ゃ' | 'ゅ' | 'ょ' => {
                let v = match c {
                    'ゃ' => "a",
                    'ゅ' => "u",
                    _ => "o",
                };
                if out.ends_with("shi") || out.ends_with("chi") || out.ends_with("ji") {
                    out.pop();
                    out.push_str(v);
                } else if out.ends_with('i') {
                    out.pop();
                    out.push('y');
                    out.push_str(v);
                } else {
                    out.push('y');
                    out.push_str(v);
                }
            }
            'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' if out.ends_with(['u', 'i']) && out.len() > 1 => {
                out.pop();
                out.push_str(T.iter().find(|t| t.0 == c).map_or("", |t| t.1));
            }
            _ => {
                let Some(&(_, r)) = T.iter().find(|t| t.0 == c) else { continue };
                if c == 'ん' {
                    out.push('n');
                    if chars.get(i + 1).is_some_and(|n| "あいうえおやゆよ".contains(*n)) {
                        out.push('\'');
                    }
                    continue;
                }
                if sokuon {
                    if r.starts_with("ch") {
                        out.push('t');
                    } else if let Some(f) = r.chars().next().filter(|f| !"aiueo".contains(*f)) {
                        out.push(f);
                    }
                    sokuon = false;
                }
                // ou / uu -> ō / ū (Hepburn); keep the plain spelling sometimes
                if (r == "u" && (out.ends_with('o') || out.ends_with('u'))) && i % 3 != 0 {
                    let l = out.pop().unwrap_or('u');
                    out.push(macron(l));
                    continue;
                }
                out.push_str(r);
            }
        }
    }
    out
}

/// Cut `s` to at most `max_chars` characters, at a space when the script has them.
pub fn clip(s: &str, max_chars: usize, by_char: bool) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.trim().to_string();
    }
    let mut end = max_chars.max(1);
    if !by_char
        && let Some(p) = chars[..end].iter().rposition(|&c| c == ' ')
            && p > 0 {
                end = p;
            }
    chars[..end].iter().collect::<String>().trim().to_string()
}

pub struct TextGen<'a> {
    pub sampler: &'a TextSampler,
    en: Vec<&'static str>,
}

impl<'a> TextGen<'a> {
    pub fn new(sampler: &'a TextSampler) -> Self {
        TextGen { sampler, en: en_words() }
    }

    fn en_word(&self, rng: &mut Rng) -> String {
        if rng.chance(0.06) { rng.pick(NAMES).to_string() } else { self.en[rng.below(self.en.len())].to_string() }
    }

    fn en_sentence(&self, rng: &mut Rng, lo: i64, hi: i64) -> String {
        let n = rng.int(lo, hi + 1) as usize;
        let mut w: Vec<String> = (0..n).map(|_| self.en_word(rng)).collect();
        if n > 5 && rng.chance(0.2) {
            let i = rng.int(1, n as i64 - 1) as usize;
            w[i].push(',');
        }
        if rng.chance(0.08) {
            let i = rng.below(n);
            w[i] = format!("\"{}\"", w[i]);
        } else if rng.chance(0.08) {
            let i = rng.below(n);
            w.insert(i, rng.pick(&["(see below)", "(1998)", "(2011)", "(right)", "(HxH)", "–", "(fig. 2)", "(n.)"]).to_string());
        }
        let mut s = cap(&w.join(" "));
        s.push_str(rng.pick(&[".", ".", ".", ".", ".", "!", "?", ":", "..."]));
        s
    }

    fn romaji_words(&self, rng: &mut Rng, n: usize) -> String {
        let mut out = Vec::new();
        for _ in 0..n {
            let k = self.sampler.sample(rng, 1, 7);
            for w in romaji(&k).split(' ') {
                if !w.is_empty() {
                    out.push(w.to_string());
                }
            }
        }
        out.join(" ")
    }

    fn ja_sentence(&self, rng: &mut Rng) -> String {
        let n = |rng: &mut Rng| if rng.chance(0.3) { rng.pick(JA_K) } else { rng.pick(JA_N) };
        let mut s = match rng.below(7) {
            0 => format!("{}は{}の{}{}", n(rng), n(rng), n(rng), rng.pick(JA_E)),
            1 => format!("{}は{}で{}。", n(rng), n(rng), rng.pick(JA_V)),
            2 => format!("{}を{}。", n(rng), rng.pick(JA_V)),
            3 => format!("この{}は{}に{}。", n(rng), n(rng), rng.pick(JA_V)),
            4 => format!("「{}」は{}{}", n(rng), n(rng), rng.pick(JA_E)),
            5 => format!("{}{}{}{}{}、{}{}{}。", n(rng), rng.pick(JA_P), n(rng), rng.pick(JA_P), rng.pick(JA_V), n(rng), rng.pick(JA_P), rng.pick(JA_V)),
            _ => format!("{}年{}月に{}が{}。", rng.int(1985, 2027), rng.int(1, 13), n(rng), rng.pick(JA_V)),
        };
        if rng.chance(0.08) {
            let ins = rng.pick(&["（Hunter × Hunter）", "（ハンター文字）", "(HxH)", "（hantā moji）", "（第36巻）"]);
            let cut = s.char_indices().nth(s.chars().count() / 2).map_or(0, |x| x.0);
            s.insert_str(cut, ins);
        }
        s
    }

    fn kana_line(&self, rng: &mut Rng, hi: usize) -> String {
        let s = self.sampler.sample(rng, 2, hi.max(2));
        let s = if rng.chance(0.4) { kata(&s) } else { s };
        if rng.chance(0.3) { format!("{s}{}", rng.pick(&["。", "、", "！", "？", "…", "ー"])) } else { s }
    }

    fn words(rng: &mut Rng, list: &[&str], lo: i64, hi: i64) -> String {
        (0..rng.int(lo, hi + 1)).map(|_| rng.pick(list)).collect::<Vec<_>>().join(" ")
    }

    fn url(rng: &mut Rng) -> String {
        let mut s = String::from(rng.pick(&["https://", "https://", "http://", "", "www.", "https://www."]));
        s.push_str(rng.pick(DOMAINS));
        s.push_str(rng.pick(TLDS));
        for _ in 0..rng.int(0, 3) {
            let p = rng.pick(PATHS);
            if !(s.ends_with('/') && p.starts_with('/')) {
                s.push_str(p);
            }
        }
        s
    }

    fn digits(rng: &mut Rng) -> String {
        let d = |rng: &mut Rng, lo: i64, hi: i64| rng.int(lo, hi + 1);
        match rng.below(16) {
            0 => format!("{}", d(rng, 1, 9999)),
            1 => format!("{}-{:02}-{:02}", d(rng, 1985, 2027), d(rng, 1, 12), d(rng, 1, 28)),
            2 => format!("{:02}/{:02}/{}", d(rng, 1, 12), d(rng, 1, 28), d(rng, 1985, 2027)),
            3 => format!("{} {}, {}", rng.pick(MONTHS), d(rng, 1, 28), d(rng, 1985, 2027)),
            4 => format!("{:02}:{:02}", d(rng, 0, 23), d(rng, 0, 59)),
            5 => format!("{}{}.{:02}", rng.pick(&["$", "€", "£"]), d(rng, 1, 999), d(rng, 0, 99)),
            6 => format!("¥{},{:03}", d(rng, 1, 99), d(rng, 0, 999)),
            7 => format!("{}%", d(rng, 1, 100)),
            8 => format!("0{}-{:04}-{:04}", d(rng, 3, 9), d(rng, 0, 9999), d(rng, 0, 9999)),
            9 => format!("{} {}", rng.pick(&["No.", "Vol.", "p.", "Ch.", "Ep.", "#", "pp."]), d(rng, 1, 400)),
            10 => format!("v{}.{}.{}", d(rng, 0, 9), d(rng, 0, 20), d(rng, 0, 99)),
            11 => format!("{}–{}", d(rng, 1985, 2010), d(rng, 2011, 2026)),
            12 => format!("978-4-08-{:06}-{}", d(rng, 0, 999_999), d(rng, 0, 9)),
            13 => format!("{} x {}", d(rng, 1, 2000), d(rng, 1, 2000)),
            14 => format!("({})", d(rng, 1985, 2027)),
            _ => format!("{}.{}", d(rng, 0, 999), d(rng, 0, 99)),
        }
    }

    /// One piece of text in `script` for `role`, at most about `max_chars` characters.
    pub fn text(&self, rng: &mut Rng, script: Script, role: Role, max_chars: usize) -> String {
        let by_char = script.by_char();
        let s = match (script, role) {
            (Script::Url, _) => Self::url(rng),
            (Script::Digits, _) => Self::digits(rng),
            (_, Role::Paragraph) => {
                let mut p = String::new();
                while p.chars().count() < max_chars {
                    let t = self.text(rng, script, Role::Sentence, 200);
                    if !p.is_empty() && !by_char {
                        p.push(' ');
                    }
                    p.push_str(&t);
                }
                return clip(&p, max_chars, by_char);
            }
            (Script::Latin, Role::Sentence) => self.en_sentence(rng, 3, 14),
            (Script::Latin, Role::Phrase) => title(&(0..rng.int(1, 4)).map(|_| self.en_word(rng)).collect::<Vec<_>>().join(" ")),
            (Script::Latin, Role::Heading) => {
                let h = title(&(0..rng.int(1, 6)).map(|_| self.en_word(rng)).collect::<Vec<_>>().join(" "));
                if rng.chance(0.2) { h.to_uppercase() } else { h }
            }
            (Script::Latin, Role::Sfx) => rng.pick(SFX_LATIN).to_string(),
            (Script::Latin, Role::Label) => match rng.below(5) {
                0 => format!("© {} {}", rng.int(1998, 2027), rng.pick(&["POT", "Shueisha", "Togashi", "Madhouse", "VIZ Media", "Nippon Animation"])),
                1 => format!("{} {}", rng.pick(&["Page", "Chapter", "Episode", "Vol.", "Fig.", "Table", "Step"]), rng.int(1, 400)),
                _ => title(&(0..rng.int(1, 3)).map(|_| self.en_word(rng)).collect::<Vec<_>>().join(" ")),
            },
            (Script::Latin, Role::Word) => cap(&self.en_word(rng)),
            (Script::Latin, Role::Link) => rng.pick(LINKS).to_string(),
            (Script::Romaji, Role::Sentence) => {
                let k = rng.int(2, 6) as usize;
                let s = cap(&self.romaji_words(rng, k));
                if rng.chance(0.5) { s + "." } else { s }
            }
            (Script::Romaji, Role::Sfx) => rng.pick(SFX_ROMAJI).to_string(),
            (Script::Romaji, Role::Heading) => {
                let k = rng.int(1, 3) as usize;
                title(&self.romaji_words(rng, k))
            }
            (Script::Romaji, _) => {
                let s = self.romaji_words(rng, 1);
                if rng.chance(0.5) { cap(&s) } else { s }
            }
            (Script::Japanese, Role::Sentence) => self.ja_sentence(rng),
            (Script::Japanese, Role::Heading) => {
                if rng.chance(0.6) {
                    rng.pick(JA_HEAD).to_string()
                } else {
                    format!("{}の{}", rng.pick(JA_K), rng.pick(JA_N))
                }
            }
            (Script::Japanese, Role::Sfx) => rng.pick(SFX_KANA).to_string(),
            (Script::Japanese, Role::Label) => match rng.below(4) {
                0 => format!("第{}話", rng.int(1, 149)),
                1 => format!("{}巻", rng.int(1, 38)),
                2 => format!("{}年{}月{}日", rng.int(1998, 2027), rng.int(1, 13), rng.int(1, 29)),
                _ => rng.pick(JA_LABEL).to_string(),
            },
            (Script::Japanese, _) => {
                if rng.chance(0.5) {
                    rng.pick(JA_N).to_string()
                } else {
                    rng.pick(JA_K).to_string()
                }
            }
            (Script::Kana, Role::Sfx) => rng.pick(SFX_KANA).to_string(),
            (Script::Kana, Role::Sentence) => self.kana_line(rng, max_chars.min(14)),
            (Script::Kana, _) => self.kana_line(rng, max_chars.min(6)),
            (s @ (Script::Korean | Script::Greek | Script::Cyrillic), r) => {
                let list = match s {
                    Script::Korean => KO,
                    Script::Greek => EL,
                    _ => RU,
                };
                match r {
                    Role::Sentence => cap(&Self::words(rng, list, 3, 9)) + ".",
                    Role::Heading | Role::Phrase => cap(&Self::words(rng, list, 1, 3)),
                    Role::Sfx => Self::words(rng, list, 1, 1).to_uppercase() + "!",
                    _ => Self::words(rng, list, 1, 2),
                }
            }
            (Script::Icon, _) => String::new(),
        };
        clip(&s, max_chars.max(1), by_char)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn romaji_hepburn() {
        assert_eq!(romaji("はんたー"), "hantā");
        assert_eq!(romaji("きゃっと"), "kyatto");
        assert_eq!(romaji("しゃしん"), "shashin");
        assert_eq!(romaji("まっちゃ"), "matcha");
        assert_eq!(romaji("ほんや"), "hon'ya");
    }
}
