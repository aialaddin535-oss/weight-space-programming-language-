"""Build the Hindi/Hinglish evaluation sets.

Run:  python scripts/build_datasets.py
Writes datasets/dev.jsonl (search) and datasets/heldout.jsonl (final report only).

Rubric fields per item
  must_include   [[tok, tok, ...], ...]  alternative tokens per semantic group,
                                         ALL groups must appear in a good answer
  answer_tokens  [tok, ...]              any one marks a verifiable correct answer
  expect_script  devanagari | roman | either
  max_words / min_words, expect_bullets  instruction constraints
  must_not_include, fact_banned          things a good answer must avoid
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "datasets" / "dev.jsonl"
HELD = ROOT / "datasets" / "heldout.jsonl"


def it(id, cat, prompt, *, script="either", must=None, ans=None, meaning="", **kw):
    d = {"id": id, "category": cat, "prompt": prompt, "expect_script": script,
         "must_include": must or [], "answer_tokens": ans or [], "meaning": meaning}
    d.update(kw)
    return d


DEV_ROWS = [
    # 1. Roman Hinglish understanding -----------------------------------------
    it("d-rom-01", "roman_hinglish", "mujhe samaj nahi aya, simple me batao",
       script="roman", must=[["samaj", "samjha", "simple", "aasan"]],
       meaning="user did not understand; wants a simpler explanation", max_words=120),
    it("d-rom-02", "roman_hinglish", "iska matlab kya hai?",
       script="roman", must=[["matlab", "meaning", "arth"]],
       meaning="asks for the meaning of something just mentioned", max_words=90),
    it("d-rom-03", "roman_hinglish", "mujhe samajh nahi aaya bhai, dobara bolo",
       script="roman", must=[["dobara", "phir", "again", "simple"]],
       meaning="spelling variant of the same request; must still be understood", max_words=120),
    it("d-rom-04", "roman_hinglish", "kal kya karna hai?",
       script="roman", must=[["kal", "tomorrow", "task", "kaam"]],
       meaning="asks about tomorrow's plan", max_words=90),
    it("d-rom-05", "roman_hinglish", "abhi ruk jao, main batata hu",
       script="roman", must=[["ruko", "ruk", "ok", "theek", "wait"]],
       meaning="a stop command, not a question - must not answer with a lecture", max_words=60),
    it("d-rom-06", "roman_hinglish", "aisa kyu ho raha hai?",
       script="roman", must=[["kyu", "kyun", "because", "wajah", "reason"]],
       meaning="asks for a cause", max_words=120),
    it("d-rom-07", "roman_hinglish", "mene pehle kya bola tha, yaad hai?",
       script="roman", must=[["pehle", "earlier", "yaad", "pichle", "last"]],
       meaning="refers back to earlier conversation; must not invent a quote", max_words=90,
       fact_banned=["you said that you are", "I never said"]),

    # 2. Devanagari Hindi ----------------------------------------------------
    it("d-dev-01", "devanagari_hindi", "मुझे समझ नहीं आया। आसान भाषा में बताइए।",
       script="devanagari", must=[["समझ", "आसान", "सरल"]],
       meaning="did not understand; wants simpler Hindi", max_words=120),
    it("d-dev-02", "devanagari_hindi", "इसका मतलब क्या है?",
       script="devanagari", must=[["मतलब", "अर्थ"]], meaning="asks for meaning", max_words=90),
    it("d-dev-03", "devanagari_hindi", "मेरे लिए एक छोटा ईमेल लिखो।",
       script="devanagari", must=[["ईमेल", "email", "विषय", "नमस्ते"]],
       meaning="asks to draft a short email", min_words=12),
    it("d-dev-04", "devanagari_hindi", "ऐसा क्यों हो रहा है?",
       script="devanagari", must=[["क्यों", "कारण", "वजह"]], meaning="asks for a cause", max_words=120),
    it("d-dev-05", "devanagari_hindi", "इसे ठीक कैसे करूँ?",
       script="devanagari", must=[["ठीक", "करें", "करूँ", "समाधान", "step"]],
       meaning="asks for a fix", max_words=150),
    it("d-dev-06", "devanagari_hindi", "मैंने पहले क्या कहा था?",
       script="devanagari", must=[["पहले", "याद", "पिछली"]],
       meaning="recall request; must not fabricate", max_words=90),
    it("d-dev-07", "devanagari_hindi", "यह क्या कर रहा है? समझाइए।",
       script="devanagari", must=[["कर रहा", "काम", "समझ"]], meaning="asks what something does", max_words=120),

    # 3. Mixed Hindi-English --------------------------------------------------
    it("d-mix-01", "mixed_code_switch", "Mera laptop slow chal raha hai, kya karu?",
       must=[["slow", "स्लो", "dheema", "धीमा"], ["clean", "storage", "memory", "startup", "update"]],
       meaning="tech support question in mixed language", max_words=160),
    it("d-mix-02", "mixed_code_switch", "Is code mein error kyu aa raha hai?",
       must=[["error", "एरर", "गलती"], ["check", "line", "log", "debug"]],
       meaning="debugging help, mixed", max_words=160),
    it("d-mix-03", "mixed_code_switch", "Mujhe API key configure karni hai.",
       must=[["api", "key", "configure", "env"]],
       meaning="wants setup steps, not a definition", max_words=160),
    it("d-mix-04", "mixed_code_switch", "Ye file download karke extract kar do.",
       must=[["download", "extract", "unzip", "command"]],
       meaning="an instruction, must give the actual steps", min_words=8),
    it("d-mix-05", "mixed_code_switch", "Iska simple explanation de.",
       must=[["simple", "आसान", "matlab", "example"]], meaning="wants simple explanation", max_words=120),
    it("d-mix-06", "mixed_code_switch", "Bhai is error ka matlab samjha de, jaldi.",
       must=[["matlab", "error", "reason", "क्यों"]], meaning="urgent, informal, wants cause", max_words=120),
    it("d-mix-07", "mixed_code_switch", "Meeting 5 baje hai, reminder set kar do.",
       must=[["reminder", "5", "पांच", "set"]], meaning="wants a reminder action", max_words=90),

    # 4. Context / follow-up --------------------------------------------------
    it("d-ctx-01", "context_followup", "pehle wala option better hai kya?",
       must=[["option", "better", "haan", "nahi", "depends", "pehla"]],
       meaning="follow-up referring to a previous option", max_words=120),
    it("d-ctx-02", "context_followup", "usko thoda simple karke bata",
       must=[["simple", "आसान", "short", "chhota"]],
       meaning="'usko' refers to the previous explanation", max_words=120),
    it("d-ctx-03", "context_followup", "isse kya fayda hoga?",
       must=[["fayda", "benefit", "फायदा", "use"]], meaning="'isse' = the previous thing", max_words=120),
    it("d-ctx-04", "context_followup", "wahi karo jo tune bola tha",
       must=[["theek", "kar", "ok", "confirm", "wahi"]],
       meaning="confirmation referencing earlier instruction", max_words=80),
    it("d-ctx-05", "context_followup", "iska matlab ye hai ki main galat tha?",
       must=[["galat", "nahi", "matlab", "सही"]],
       meaning="seeks confirmation, not a translation", max_words=110,
       must_not_include=["here is the translation"]),
    it("d-ctx-06", "context_followup", "aur uske baad?",
       must=[["baad", "next", "फिर", "step"]], meaning="what happens after that", max_words=100),
    it("d-ctx-07", "context_followup", "haan theek hai, aage badho",
       must=[["theek", "aage", "ok", "next"]], meaning="approval to continue", max_words=70),

    # 5. Instruction following ------------------------------------------------
    it("d-ins-01", "instruction_following", "3 bullet points me batao ki Python kyu seekhe",
       must=[["python"], ["bullet", "-", "•", "1."]],
       meaning="exactly 3 bullets, short", expect_bullets=3, max_words=90),
    it("d-ins-02", "instruction_following", "Do line me hi jawab do: HTTP kya hai?",
       max_words=45, must=[["http", "protocol", "request", "web"]],
       meaning="hard length limit", min_words=5),
    it("d-ins-03", "instruction_following", "Sirf ek number likho: 17 + 25 kitna hai?",
       ans=["42"], must=[["42"]], max_words=8, meaning="numeric only", ),
    it("d-ins-04", "instruction_following", "English me mat likhna, sirf Hindi me: नमस्ते का reply",
       script="devanagari", must=[["नमस्ते", "नमस्कार", "हैलो"]],
       meaning="language constraint must be obeyed", max_words=40),
    it("d-ins-05", "instruction_following", "Ye paragraph 2 line me summarize karo: GST ek indirect tax hai jo India me 2017 se lagu hua.",
       max_words=50, must=[["gst", "tax", "indirect", "2017"]], meaning="summary constraint", min_words=5),
    it("d-ins-06", "instruction_following", "List format me 2 steps do: password change kaise kare",
       expect_bullets=2, must=[["password", "settings", "change", "reset"]],
       meaning="list format required", max_words=80),
    it("d-ins-07", "instruction_following", "Bina example diye, sirf definition do: machine learning",
       must=[["data", "learn", "model", "algorithm"]], max_words=60,
       meaning="no examples allowed", must_not_include=["for example"]),

    # 6. Question answering ---------------------------------------------------
    it("d-qa-01", "question_answering", "Bharat ki rajdhani kya hai?",
       must=[["delhi", "दिल्ली"]], ans=["delhi", "दिल्ली"], meaning="simple factual", max_words=25),
    it("d-qa-02", "question_answering", "2 + 2 * 3 kitna hota hai?",
       ans=["8", "आठ"], must=[["8", "आठ"]], meaning="operator precedence", max_words=25),
    it("d-qa-03", "question_answering", "HTTP aur HTTPS me kya difference hai?",
       must=[["secure", "encrypt", "ssl", "tls", "https"]], meaning="technical comparison", max_words=120),
    it("d-qa-04", "question_answering", "Python me list aur tuple me kya farak hai?",
       must=[["mutable", "immutable", "list", "tuple"]], meaning="programming QA", max_words=140),
    it("d-qa-05", "question_answering", "Ek GB me kitne MB hote hain?",
       ans=["1024", "1000"], must=[["1024", "1000"]], meaning="unit conversion", max_words=25),
    it("d-qa-06", "question_answering", "Samay ka sabse bada unit kya hai?",
       must=[["kalpa", "yuga", "era", "parsec", "universe", "year"]], meaning="open factual", max_words=60),
    it("d-qa-07", "question_answering", "CPU aur GPU me basic difference batao",
       must=[["cpu", "gpu", "core", "parallel"]], meaning="hardware QA", max_words=120),

    # 7. Natural response quality --------------------------------------------
    it("d-nat-01", "natural_response", "Yaar aaj bahut thak gaya hu, kuch motivation do",
       must=[["thak", "rest", "aaram", "kal", "good"]], meaning="casual chat, needs a human reply",
       max_words=120, must_not_include=["as an ai"]),
    it("d-nat-02", "natural_response", "Bhai maza aa gaya, thanks!",
       must=[["thanks", "khushi", "swagat", "welcome", "shukriya"]],
       meaning="small talk, must not turn into a lecture", max_words=50),
    it("d-nat-03", "natural_response", "Kal chutti hai kya?",
       must=[["chutti", "holiday", "leave", "nahi", "-"]],
       meaning="casual question; a clarifying question is acceptable", max_words=70),
    it("d-nat-04", "natural_response", "Mujhe neend aa rahi hai par kaam bacha hai",
       must=[["so", "rest", "break", "kaam", "timer"]], meaning="empathic, practical advice", max_words=100),
    it("d-nat-05", "natural_response", "Tum kaun ho?",
       must=[["model", "assistant", "AI", "madad", "help"]],
       meaning="identity question, short answer", max_words=60, must_not_include=["as an ai language model"]),
    it("d-nat-06", "natural_response", "Aaj mood kharab hai, kya karu",
       must=[["mood", "baat", "walk", "break", "suno"]],
       meaning="emotional support, informal", max_words=110),
    it("d-nat-07", "natural_response", "Chalo phir baad me baat karte hain",
       must=[["theek", "baad", "ok", "bye", "phir"]], meaning="conversation close", max_words=50),

    # 8. Spelling / variation robustness -------------------------------------
    it("d-sp-01", "spelling_robustness", "mujhe smjh nhi aya",
       must=[["samaj", "simple", "aasan", "matlab"]], meaning="vowel-dropped spelling", max_words=110),
    it("d-sp-02", "spelling_robustness", "ye kya kr rha hai",
       must=[["kar", "kya", "kaam", "samaj"]], meaning="no punctuation, no vowels", max_words=110),
    it("d-sp-03", "spelling_robustness", "iska mtlb kya h",
       must=[["matlab", "meaning", "arth"]], meaning="abbreviated spelling", max_words=90),
    it("d-sp-04", "spelling_robustness", "muje email likhna h ofis ke liye",
       must=[["email", "ईमेल", "office", "subject"]], meaning="typos + missing punctuation", max_words=140),
    it("d-sp-05", "spelling_robustness", "plz jaldi batao error kyu aya",
       must=[["error", "kyu", "karan", "check"]], meaning="informal urgency", max_words=110),
    it("d-sp-06", "spelling_robustness", "kya h ye? smjh ni aya",
       must=[["kya", "samaj", "explain", "matlab"]], meaning="double abbreviation", max_words=110),
    it("d-sp-07", "spelling_robustness", "MERA LAPTOP SLOW HAI KYA KRU",
       must=[["slow", "memory", "startup", "clean", "update"]], meaning="all caps frustration", max_words=120),

    # 9. Technical Hindi/Hinglish --------------------------------------------
    it("d-tec-01", "technical_hindi", "Docker container aur virtual machine me difference samjhao",
       must=[["docker", "container", "vm", "virtual", "os"]], meaning="technical explainer in Hinglish", max_words=180),
    it("d-tec-02", "technical_hindi", "Git rebase kya karta hai, simple Hindi me",
       must=[["rebase", "commit", "history", "branch"]], meaning="technical, simple Hindi", max_words=160),
    it("d-tec-03", "technical_hindi", "REST API me GET aur POST ka farak",
       must=[["get", "post", "data", "request", "server"]], meaning="API concept", max_words=140),
    it("d-tec-04", "technical_hindi", "Model weights kya hote hain?",
       must=[["weight", "parameter", "number", "tensor", "network"]], meaning="ML concept", max_words=140),
    it("d-tec-05", "technical_hindi", "Database index kaise fast karta hai",
       must=[["index", "search", "b-tree", "lookup", "fast"]], meaning="DB concept", max_words=150),
    it("d-tec-06", "technical_hindi", "SSH key kaise banate hain, command ke saath",
       must=[["ssh-keygen", "ssh", "key", "command"]], meaning="needs an actual command", max_words=120),
    it("d-tec-07", "technical_hindi", "Environment variable kya hota hai, example ke saath",
       must=[["environment", "variable", "export", "value"]], meaning="needs an example", max_words=130),

    # 10. Factuality ----------------------------------------------------------
    it("d-fac-01", "factuality", "Bharat ke pradhan mantri kaun hain?",
       must=[["modi", "मोदी", "narendra"]], meaning="current-office question; must not invent",
       fact_banned=["rahul gandhi is the prime minister"], max_words=40),
    it("d-fac-02", "factuality", "Qwen3.5-0.8B model kitne parameters ka hai?",
       must=[["0.8", "800", "0.8b"]], ans=["0.8", "800m"], meaning="model-size question", max_words=50),
    it("d-fac-03", "factuality", "UN ka headquarter kahan hai?",
       must=[["new york", "न्यूयॉर्क"]], ans=["new york", "न्यूयॉर्क"],
       meaning="stable fact", max_words=30),
    it("d-fac-04", "factuality", "2024 Olympics kahan hue the?",
       must=[["paris", "पेरिस"]], ans=["paris", "पेरिस"], meaning="recent fact", max_words=30),
    it("d-fac-05", "factuality", "Internet ka invention kisne kiya?",
       must=[["berners-lee", "tcp", "arpanet", "cerf", "kahn"]],
       meaning="accepts multiple valid answers, must not invent one name", max_words=80,
       fact_banned=["bill gates invented the internet"]),
    it("d-fac-06", "factuality", "Hindi me kitne vowels hote hain?",
       must=[["11", "ग्यारह", "swar", "स्वर"]], ans=["11", "ग्यारह"],
       meaning="counting question", max_words=40),
    it("d-fac-07", "factuality", "Ek saal me kitne din hote hain?",
       must=[["365", "366"]], ans=["365"], meaning="stable fact", max_words=25),
]

HELD_ROWS = [
    it("h-rom-01", "roman_hinglish", "bhai simple language me samjha de",
       script="roman", must=[["simple", "aasan", "matlab"]], meaning="held-out Roman Hinglish", max_words=120),
    it("h-rom-02", "roman_hinglish", "isse thik kaise karu",
       script="roman", must=[["thik", "theek", "fix", "kar"]], meaning="held-out fix request", max_words=140),
    it("h-rom-03", "roman_hinglish", "mujhe samajh nahi aaya phir se bolo",
       script="roman", must=[["phir", "dobara", "simple"]], meaning="held-out repetition request", max_words=120),
    it("h-rom-04", "roman_hinglish", "abhi mat karo, baad me dekhte hain",
       script="roman", must=[["baad", "ok", "theek", "later"]], meaning="held-out deferral", max_words=60),

    it("h-dev-01", "devanagari_hindi", "मुझे समझ नहीं आया, फिर से बताइए।",
       script="devanagari", must=[["समझ", "फिर", "दोबारा"]], meaning="held-out Devanagari", max_words=120),
    it("h-dev-02", "devanagari_hindi", "इसका सरल अर्थ क्या है?",
       script="devanagari", must=[["अर्थ", "मतलब", "सरल"]], meaning="held-out meaning", max_words=100),
    it("h-dev-03", "devanagari_hindi", "यह त्रुटि क्यों आ रही है?",
       script="devanagari", must=[["त्रुटि", "कारण", "क्यों"]], meaning="held-out error question", max_words=140),
    it("h-dev-04", "devanagari_hindi", "मेरे लिए दो पंक्तियों का संदेश लिखो।",
       script="devanagari", must=[["संदेश", "message", "नमस्ते"]], meaning="held-out drafting", min_words=6),

    it("h-mix-01", "mixed_code_switch", "Mera phone hang ho raha hai, kya karu?",
       must=[["restart", "storage", "app", "update"]], meaning="held-out tech support", max_words=150),
    it("h-mix-02", "mixed_code_switch", "Is script ka output kya aayega?",
       must=[["output", "result", "print"]], meaning="held-out code question", max_words=140),
    it("h-mix-03", "mixed_code_switch", "Mujhe database backup lena hai, steps do.",
       must=[["backup", "database", "command", "step"]], meaning="held-out ops request", min_words=8),
    it("h-mix-04", "mixed_code_switch", "Ye link kaam nahi kar raha, kya issue hai?",
       must=[["link", "url", "check", "issue"]], meaning="held-out troubleshooting", max_words=140),

    it("h-ctx-01", "context_followup", "uska alternative kya hai?",
       must=[["alternative", "doosra", "option"]], meaning="held-out reference", max_words=110),
    it("h-ctx-02", "context_followup", "iske baad kya karna chahiye?",
       must=[["baad", "next", "step"]], meaning="held-out sequencing", max_words=120),
    it("h-ctx-03", "context_followup", "wahi wala example dobara do",
       must=[["example", "udaharan", "wahi"]], meaning="held-out recall", max_words=120),
    it("h-ctx-04", "context_followup", "haan wahi karo, confirm",
       must=[["confirm", "theek", "ok"]], meaning="held-out confirmation", max_words=60),

    it("h-ins-01", "instruction_following", "2 bullet me batao: laptop slow kyu hota hai",
       expect_bullets=2, must=[["laptop", "slow", "memory", "startup"]], meaning="held-out bullets", max_words=70),
    it("h-ins-02", "instruction_following", "Sirf ek shabd me batao: aasman ka rang?",
       ans=["neela", "nila", "blue", "नीला"], must=[["neela", "nila", "blue", "नीला"]],
       meaning="held-out one-word constraint", max_words=5),
    it("h-ins-03", "instruction_following", "Hindi me jawab do: what is a variable?",
       script="devanagari", must=[["चर", "variable", "मान"]], meaning="held-out language constraint", max_words=80),
    it("h-ins-04", "instruction_following", "3 step list do: naya project setup",
       expect_bullets=3, must=[["step", "setup", "folder", "install"]], meaning="held-out list", max_words=90),

    it("h-qa-01", "question_answering", "Bharat me kitne rajya hain?",
       ans=["28", "अट्ठाईस"], must=[["28", "अट्ठाईस"]], meaning="held-out fact", max_words=30),
    it("h-qa-02", "question_answering", "9 * 9 kitna hai?",
       ans=["81"], must=[["81"]], meaning="held-out arithmetic", max_words=20),
    it("h-qa-03", "question_answering", "RAM aur ROM me farak batao",
       must=[["ram", "rom", "volatile", "memory"]], meaning="held-out hardware", max_words=120),
    it("h-qa-04", "question_answering", "JSON kya hai, ek line me",
       must=[["json", "data", "format", "key"]], meaning="held-out definition", max_words=50),

    it("h-nat-01", "natural_response", "Aaj bahut bura din tha",
       must=[["suno", "baat", "din", "hope", "kal"]], meaning="held-out empathy", max_words=110,
       must_not_include=["as an ai"]),
    it("h-nat-02", "natural_response", "Thanks bhai, kaam ho gaya",
       must=[["thanks", "shukriya", "khushi", "welcome"]], meaning="held-out thanks", max_words=50),
    it("h-nat-03", "natural_response", "Thoda time chahiye, kal baat karein?",
       must=[["kal", "theek", "ok", "time"]], meaning="held-out scheduling", max_words=70),
    it("h-nat-04", "natural_response", "Tumse baat karke accha laga",
       must=[["accha", "khushi", "shukriya", "thanks"]], meaning="held-out small talk", max_words=60),

    it("h-sp-01", "spelling_robustness", "mujhe smj ni aa rha kya krna h",
       must=[["samaj", "simple", "step", "matlab"]], meaning="held-out typos", max_words=120),
    it("h-sp-02", "spelling_robustness", "ye err kyu aa rha h",
       must=[["error", "kyu", "reason", "check"]], meaning="held-out abbreviated", max_words=120),
    it("h-sp-03", "spelling_robustness", "plzzzz jldi btao",
       must=[["jaldi", "bata", "ok", "theek"]], meaning="held-out stretched spelling", max_words=80),
    it("h-sp-04", "spelling_robustness", "MERE KO SMJH NHI AYA",
       must=[["samaj", "simple", "matlab"]], meaning="held-out caps + typos", max_words=110),

    it("h-tec-01", "technical_hindi", "Kubernetes pod aur deployment me farak",
       must=[["pod", "deployment", "kubernetes", "replica"]], meaning="held-out k8s", max_words=170),
    it("h-tec-02", "technical_hindi", "Transformer architecture simple Hindi me samjhao",
       must=[["attention", "transformer", "layer", "token"]], meaning="held-out ML", max_words=170),
    it("h-tec-03", "technical_hindi", "Cron job kya hota hai, cron syntax batao",
       must=[["cron", "schedule", "*", "job"]], meaning="held-out ops", max_words=140),
    it("h-tec-04", "technical_hindi", "SSL certificate expire hone ka kya matlab hai",
       must=[["ssl", "certificate", "expire", "browser", "https"]], meaning="held-out security", max_words=150),

    it("h-fac-01", "factuality", "Taj Mahal kahan hai?",
       ans=["agra", "आगरा"], must=[["agra", "आगरा"]], meaning="held-out fact", max_words=25),
    it("h-fac-02", "factuality", "Ek hour me kitne seconds hote hain?",
       ans=["3600"], must=[["3600"]], meaning="held-out computation", max_words=25),
    it("h-fac-03", "factuality", "Python ka inventor kaun hai?",
       ans=["guido", "van rossum"], must=[["guido", "rossum"]], meaning="held-out fact", max_words=40),
    it("h-fac-04", "factuality", "Bharat ka national animal kaun hai?",
       ans=["tiger", "बाघ"], must=[["tiger", "बाघ"]], meaning="held-out fact", max_words=25),
]


# ---------------------------------------------------------------------------
# V4 required capability probes.
#
# The specification names these exact messages as the behaviours Aladdin WSP
# exists to improve. They are split across dev (search) and held-out (final
# report only); no prompt appears in both.
# ---------------------------------------------------------------------------
DEV_ROWS += [
    it("d-spec-01", "roman_hinglish", "mujhe ye samjha",
       script="roman", must=[["samaj", "samjha", "matlab", "simple"]],
       meaning="short Roman Hinglish 'explain this to me'; spelling variant", max_words=120),
    it("d-spec-02", "roman_hinglish", "iska kya matlab hai",
       script="roman", must=[["matlab", "meaning", "arth"]],
       meaning="asks what the previous thing means", max_words=90),
    it("d-spec-03", "context_followup", "ab isko thik karo",
       must=[["thik", "theek", "fix", "kar"]],
       meaning="'isko' refers to the previous item; wants a fix", max_words=120),
    it("d-spec-04", "instruction_following", "mere liye ek accha assistant banao",
       must=[["assistant", "assist", "help", "madad", "banao"]],
       meaning="asks for an assistant to be built", max_words=140),
    it("d-spec-05", "devanagari_hindi", "मुझे यह समझाओ",
       script="devanagari", must=[["समझ", "बताओ", "समझाओ"]],
       meaning="Devanagari 'explain this to me'", max_words=120),
]

HELD_ROWS += [
    it("h-spec-01", "spelling_robustness", "muje ye samja",
       must=[["samaj", "samjha", "matlab", "simple"]],
       meaning="held-out misspelling of 'mujhe ye samjha'", max_words=120),
    it("h-spec-02", "context_followup", "kal wala kaam continue karo",
       must=[["kaam", "continue", "aage", "kal", "task"]],
       meaning="held-out reference to yesterday's task; continue it", max_words=130),
    it("h-spec-03", "devanagari_hindi", "इसका मतलब क्या है",
       script="devanagari", must=[["मतलब", "अर्थ"]],
       meaning="held-out Devanagari meaning question", max_words=90),
    it("h-spec-04", "context_followup", "अब इसे ठीक करो",
       script="devanagari", must=[["ठीक", "करो", "समाधान", "fix"]],
       meaning="held-out Devanagari pronoun reference + fix request", max_words=120),
    it("h-spec-05", "mixed_code_switch", "मुझे एक अच्छा AI assistant बनाना है",
       must=[["assistant", "ai", "बनाना", "help", "मदद"]],
       meaning="held-out Hindi-English mixed: wants a good AI assistant built", max_words=140),
]


def main() -> None:
    for path, rows, split in ((DEV, DEV_ROWS, "dev"), (HELD, HELD_ROWS, "heldout")):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cats = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(f"{path.relative_to(ROOT)}: {len(rows)} items, {len(cats)} categories -> {cats}")

    # integrity: no shared ids, no shared prompts
    dev_ids = {r["id"] for r in DEV_ROWS}
    held_ids = {r["id"] for r in HELD_ROWS}
    assert not (dev_ids & held_ids), "id overlap"
    p_dev = {r["prompt"] for r in DEV_ROWS}
    p_held = {r["prompt"] for r in HELD_ROWS}
    assert not (p_dev & p_held), f"prompt overlap: {p_dev & p_held}"
    print("split check: OK (no shared ids or prompts)")


if __name__ == "__main__":
    main()
