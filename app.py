import base64, html, io, sys, time
from pathlib import Path
import numpy as np
import gradio as gr
import pandas as pd
from PIL import Image, ImageOps

from tlcv.common import CONDITIONS, HELD_OUT, CLASSES, PHOTOS, load_split
from tlcv.pipeline import CLASSIFIERS, DETECTORS, Engine, draw, environment, heat_overlay, mask_overlay, watermark

ROOT = Path(__file__).resolve().parent
E = Engine()
DET_KEYS, CLS_KEYS = list(DETECTORS), list(CLASSIFIERS)
EXAMPLES = sorted(str(p) for p in PHOTOS.glob("*.jpg"))
EX_PICK = [EXAMPLES[i] for i in (0, 4, 13, 21, 31) if i < len(EXAMPLES)]
LAT = {"detect": [], "classify": [], "explain": []}
MODEL_SHORT = {CLS_KEYS[0]: "A2 · ResNet18 + CondSim", CLS_KEYS[1]: "A1 · ResNet18", CLS_KEYS[2]: "YOLO11s-cls"}
_test = None


class Text:
    def __init__(self, lang):
        self.ru = lang == "ru"
        L = self.L
        self.name = {"red": L("Red", "Красный"), "yellow": L("Yellow", "Жёлтый"), "green": L("Green", "Зелёный")}
        self.meaning = {"red": L("stop", "стоп"), "yellow": L("get ready", "внимание"), "green": L("go", "можно ехать")}
        self.cond = {"night": L("Night", "Ночь"), "glare": L("Glare", "Засветка"), "motion": L("Motion blur", "Смаз"),
                     "rain": L("Rain", "Дождь"), "fog": L("Fog", "Туман"), "noise+jpeg": L("Noise + JPEG", "Шум + JPEG"),
                     "low-res": L("Low resolution", "Низкое разрешение")}

    def L(self, en, ru):
        return ru if self.ru else en

    def note(self, st):
        L = self.L
        if st == "ok": return L("Confident", "Уверенно")
        if st == "no lit lamp": return L("No lit lamp found", "Горящая лампа не найдена")
        if st == "low confidence": return L("Low confidence, check by eye", "Низкая уверенность, проверьте глазами")
        c = self.name[st.split()[-1]].lower()
        return L(f"Pixel colour looks {c}, so the result is doubtful", f"По цвету пикселей похоже на «{c}», результат под вопросом")

    def ms(self, v):
        return f"{v / 1000:.1f} " + self.L("s", "с") if v >= 1000 else f"{v:.0f} " + self.L("ms", "мс")


def signal_head(label, sure=True, small=False):
    lamps = "".join(f'<span class="lamp {c}{" on" if c == label else ""}{" dim" if c == label and not sure else ""}"></span>'
                    for c in ["red", "yellow", "green"])
    return f'<div class="sighead{" sm" if small else ""}" aria-hidden="true">{lamps}</div>'


def thumbs_html(items, empty=""):
    if not items:
        return f'<div class="thumbs-empty">{empty}</div>' if empty else ""
    tiles = []
    for im, cap in items:
        im = im.convert("RGB"); im.thumbnail((360, 360)); buf = io.BytesIO(); im.save(buf, "PNG")
        tiles.append(f'<figure><img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="{html.escape(cap)}">'
                     f'<figcaption>{html.escape(cap)}</figcaption></figure>')
    return f'<div class="thumbs">{"".join(tiles)}</div>'


def card(T, k, d):
    sure = d["status"] == "ok"; lab = d["label"]; pct = round(100 * d["conf"])
    return (f'<div class="card{"" if sure else " unsure"}">{signal_head(lab, sure)}'
            f'<div class="body"><div class="row"><span class="num">{k}</span><span class="title">{T.name[lab]}{"" if sure else " ?"}</span>'
            f'<span class="meaning">{T.meaning[lab] if sure else ""}</span></div>'
            f'<div class="bar"><span class="fill {lab}" style="width:{pct}%"></span></div>'
            f'<div class="meta">{T.L("Confidence", "Уверенность")} {pct}% — {html.escape(T.note(d["status"]))}</div></div></div>')


def env_md(T, request):
    e = environment()
    ua = request.headers.get("user-agent", "") if request is not None else ""
    browser = next((b for b in ["Edg", "OPR", "Chrome", "Firefox", "Safari"] if b in ua), T.L("unknown", "неизвестно"))
    avg = lambda k: T.ms(np.mean(LAT[k])) if LAT[k] else "—"
    L = T.L
    return (f"**{L('Device', 'Устройство')}:** {e['device']}  \n**{L('System', 'Система')}:** {e['os']}, Python {e['python']}, "
            f"PyTorch {e['torch']}  \n**{L('Browser', 'Браузер')}:** {browser}  \n"
            f"**{L('Average in this session', 'Среднее за сессию')}:** {L('detection', 'детекция')} {avg('detect')}, "
            f"{L('classification per light', 'классификация на светофор')} {avg('classify')}, "
            f"{L('explanation per light', 'объяснение на светофор')} {avg('explain')}")


def analyse(T, img, det_key, cls_key, det_thr, min_conf, explain, show_unlit, request):
    L = T.L
    if img is None:
        raise gr.Error(L("Add a photo first: upload one, take one with the camera or pick an example.",
                         "Сначала добавьте фото: загрузите, снимите камерой или выберите пример."))
    img = ImageOps.exif_transpose(img).convert("RGB")
    dets, t = E.analyse(img, det_key, cls_key, det_thr, min_conf)
    LAT["detect"].append(t["detect_ms"])
    if dets: LAT["classify"].append(t["classify_ms"] / len(dets))
    shown = [d for d in dets if show_unlit or d["status"] != "no lit lamp"]
    gallery, rows = [], []
    t0 = time.perf_counter()
    for k, d in enumerate(shown, 1):
        rows.append([k, T.name[d["label"]] + ("" if d["status"] == "ok" else " ?"), f"{d['conf']:.2f}",
                     *[f"{d['probs'][c]:.2f}" for c in CLASSES], f"{d['det_score']:.2f}", T.note(d["status"])])
        if k > 6: continue
        if "Grad-CAM" in explain:
            cam = E.gradcam(cls_key, d["crop"], CLASSES.index(d["label"]))
            gallery.append((heat_overlay(d["crop"], cam), f"{k}. Grad-CAM — {T.name[d['label']]}"))
        if "SAM 2 mask" in explain:
            pb, big, house, lamp, pt = E.sam_lamp_mask(img, d["box"])
            gallery.append((mask_overlay(big, house, lamp, pt), f"{k}. SAM 2 — " + L("housing / lit lamp", "корпус / горящая лампа")))
    ex_ms = (time.perf_counter() - t0) * 1000
    if explain and shown: LAT["explain"].append(ex_ms / len(shown))
    table = pd.DataFrame(rows, columns=["#", L("Signal", "Сигнал"), L("Confidence", "Уверенность"), "p(red)", "p(yellow)",
                                        "p(green)", L("Detector score", "Оценка детектора"), L("Check", "Проверка")])
    if shown:
        timing = (f'{L("Detection", "Детекция")} {T.ms(t["detect_ms"])}, {L("classification", "классификация")} '
                  f'{T.ms(t["classify_ms"])}' + (f', {L("explanation", "объяснение")} {T.ms(ex_ms)}' if ex_ms > 1 else ""))
        cards = f'<div class="cards">{"".join(card(T, k, d) for k, d in enumerate(shown, 1))}</div><div class="timing">{timing}</div>'
    else:
        cards = (f'<div class="empty"><b>{L("No lit traffic lights found", "Горящих светофоров не найдено")}</b><br>'
                 f'{L("Lower the detector threshold in Settings or turn on “Show unlit lights”.", "Уменьшите порог детектора в настройках или включите «Показывать негорящие».")}</div>')
    empty = L("Turn on an explanation in Settings to see it here.", "Включите объяснение в настройках, чтобы увидеть его здесь.")
    return draw(img, shown, True, names=T.name), cards, thumbs_html(gallery, empty if shown else ""), table, env_md(T, request)


def random_crop(T):
    global _test
    if _test is None: _test = load_split()[2]
    r = _test.sample(1).iloc[0]
    im = Image.open(r.path).convert("RGB")
    if im.size[1] < 200: im = im.resize((max(1, round(im.size[0] * 200 / im.size[1])), 200), Image.BICUBIC)
    return im, f'{T.L("True signal", "Настоящий сигнал")}: **{T.name[r.label]}**'


def simulate(T, crop, cond, strength):
    L = T.L
    if crop is None:
        raise gr.Error(L("Pick a crop first: press “Random crop” or upload a close-up of one traffic light.",
                         "Сначала выберите кроп: нажмите «Случайный кроп» или загрузите крупный план светофора."))
    crop = crop.convert("RGB")
    big = crop.resize((max(96, crop.size[0] * 3), max(96, crop.size[1] * 3)), Image.BICUBIC) if min(crop.size) < 64 else crop
    sim = big if cond == "none" else CONDITIONS[cond](big, float(strength), np.random.default_rng(0))
    cards, gal = [], []
    for key in CLS_KEYS:
        p = E.classify(key, [sim])[0]; lab = max(p, key=p.get); pct = round(100 * p[lab])
        cards.append(f'<div class="card compact">{signal_head(lab, small=True)}<div class="body"><div class="row">'
                     f'<span class="title">{T.name[lab]}</span><span class="meaning">{pct}%</span></div>'
                     f'<div class="bar"><span class="fill {lab}" style="width:{pct}%"></span></div>'
                     f'<div class="meta">{MODEL_SHORT[key]}</div></div></div>')
        gal.append((heat_overlay(sim, E.gradcam(key, sim, CLASSES.index(lab))), MODEL_SHORT[key]))
    shown = sim if cond == "none" else watermark(sim.resize((224, int(224 * sim.size[1] / sim.size[0]))), L("SYNTHETIC", "СИНТЕТИКА"))
    note = "" if cond == "none" else (
        f'<div class="synth">{L("Synthetic image: a computer simulation of", "Синтетическое изображение: компьютерная симуляция условия")} '
        f'«{T.cond[cond]}», {L("not a real photo", "а не настоящее фото")}.'
        + (f' {L("The A2 model never saw this condition in training.", "Модель A2 не видела это условие при обучении.")}' if cond in HELD_OUT else "")
        + "</div>")
    return shown, note + f'<div class="cards three">{"".join(cards)}</div>', thumbs_html(gal)


ETHICS = {"en": """
#### Use and limits
* **This is a student prototype, not a safety system.** Never use it to decide whether to cross the road or drive. It can miss lights, and on blurred or over-exposed lamps it can be confidently wrong.
* **Synthetic images.** The simulator makes computer-generated variants (night, glare, rain, fog, blur, noise, low resolution). Each one is marked “SYNTHETIC”. No generative model is used, and no people or faces are created.
* **Your photos.** Images are processed in memory and **not saved**. Blur faces and licence plates before you share results.
* **Data limits.** The data comes from a few cities, mostly in daylight and dry weather, with few yellow examples. Real night, snow and other countries' signals are under-represented.
""", "ru": """
#### Использование и ограничения
* **Это студенческий прототип, а не система безопасности.** Не используйте его, чтобы решать, переходить ли дорогу или ехать. Он может пропустить светофор, а на размытых или пересвеченных лампах уверенно ошибиться.
* **Синтетические изображения.** Симулятор создаёт компьютерные варианты (ночь, засветка, дождь, туман, смаз, шум, низкое разрешение). Каждый помечен «СИНТЕТИКА». Генеративные модели не используются, люди и лица не создаются.
* **Ваши фото.** Изображения обрабатываются в памяти и **не сохраняются**. Размойте лица и номера машин, прежде чем делиться результатами.
* **Ограничения данных.** Данные из нескольких городов, в основном днём и в сухую погоду, жёлтых примеров мало. Настоящая ночь, снег и светофоры других стран представлены слабо.
"""}


def build(lang):
    T = Text(lang); L = T.L
    key_results = [("0.902 → 0.922", L("macro-F1 of the signal classifier (A1 → A2)", "macro-F1 классификатора сигнала (A1 → A2)")),
                   ("0.81 → 0.91", L("F1 in rain and fog never seen in training", "F1 в дождь и туман, которых не было в обучении")),
                   ("33 → 8", L("false boxes of the detector on 33 own photos", "ложных боксов детектора на 33 своих фото")),
                   ("0.53 → 0.82", L("precision of the detector at threshold 0.3", "точность (precision) детектора при пороге 0.3"))]
    gr.HTML(f'<div class="hero">{signal_head("green")}<div><h1>{L("Which signal is on?", "Какой сигнал горит?")}</h1>'
            f'<p>{L("Add a street photo. The app finds the traffic lights, reads each signal and shows what the model looked at.", "Добавьте фото улицы. Приложение найдёт светофоры, определит сигнал каждого и покажет, на что смотрела модель.")}</p></div></div>')

    with gr.Tab(L("Recognise", "Распознать")):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                inp = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label=L("Photo", "Фото"), height=380)
                btn = gr.Button(L("Find traffic lights", "Найти светофоры"), variant="primary", elem_classes="run")
                if EX_PICK:
                    gr.Examples(EX_PICK, inp, label=L("Or try one of our photos", "Или возьмите одно из наших фото"))
                with gr.Accordion(L("Settings", "Настройки"), open=False):
                    det_key = gr.Dropdown(DET_KEYS, value=DET_KEYS[0], label=L("Detector", "Детектор"))
                    cls_key = gr.Dropdown(CLS_KEYS, value=CLS_KEYS[0], label=L("Signal classifier", "Классификатор сигнала"),
                                          info=L("A2 is the final model; A1 is last assignment's model, for comparison.",
                                                 "A2 — итоговая модель; A1 — модель из прошлого задания, для сравнения."))
                    explain = gr.CheckboxGroup([(L("Heat-map (Grad-CAM)", "Тепловая карта (Grad-CAM)"), "Grad-CAM"),
                                                (L("Lamp mask (SAM 2, +2–10 s)", "Маска лампы (SAM 2, +2–10 с)"), "SAM 2 mask")],
                                               value=["Grad-CAM"], label=L("Explanations", "Объяснения"))
                    det_thr = gr.Slider(0.1, 0.9, 0.30, step=0.05, label=L("Detector threshold", "Порог детектора"),
                                        info=L("Lower finds more lights but also more false ones.", "Ниже — больше найденных светофоров, но и больше ложных."))
                    min_conf = gr.Slider(0.0, 0.95, 0.50, step=0.05, label=L("Confidence needed for a sure answer", "Уверенность для надёжного ответа"))
                    show_unlit = gr.Checkbox(False, label=L("Show unlit lights", "Показывать негорящие"))
            with gr.Column(scale=7, min_width=360):
                out_img = gr.Image(type="pil", label=L("Result", "Результат"), height=460, interactive=False)
                cards = gr.HTML(f'<div class="empty"><b>{L("Results will appear here", "Здесь появится результат")}</b><br>'
                                f'{L("Add a photo on the left and press “Find traffic lights”.", "Добавьте фото слева и нажмите «Найти светофоры».")}</div>')
        with gr.Accordion(L("Why did the model decide this?", "Почему модель так решила?"), open=True):
            gr.Markdown(L("Red areas on the heat-map are what the model relied on. A good model looks at the lit lamp. "
                          "The SAM 2 mask outlines the housing (blue) and the lit lamp (magenta).",
                          "Красные области тепловой карты — то, на что опиралась модель. Хорошая модель смотрит на горящую лампу. "
                          "Маска SAM 2 обводит корпус (синим) и горящую лампу (пурпурным)."))
            gallery = gr.HTML()
        with gr.Accordion(L("Details: probabilities, speed, device", "Подробности: вероятности, скорость, устройство"), open=False):
            table = gr.Dataframe(show_label=False, wrap=True)
            env = gr.Markdown()

        def run(img, dk, ck, thr, mc, ex, su, request: gr.Request):
            return analyse(T, img, dk, ck, thr, mc, ex, su, request)
        btn.click(run, [inp, det_key, cls_key, det_thr, min_conf, explain, show_unlit], [out_img, cards, gallery, table, env])

    with gr.Tab(L("Bad-weather simulator", "Симулятор погоды")):
        gr.Markdown(L("See how the models cope with bad conditions. Take a close-up of one traffic light, add night, glare, rain or fog, "
                      "and compare the answers of the A2 model, the A1 model and YOLO11s-cls on exactly the same image.",
                      "Посмотрите, как модели справляются с плохими условиями. Возьмите крупный план одного светофора, добавьте ночь, "
                      "засветку, дождь или туман и сравните ответы модели A2, модели A1 и YOLO11s-cls на одном и том же изображении."))
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                crop = gr.Image(type="pil", label=L("Traffic-light close-up", "Крупный план светофора"), height=230)
                crop_info = gr.Markdown()
                gr.Button(L("Random crop from the test set", "Случайный кроп из теста")).click(lambda: random_crop(T), None, [crop, crop_info])
                cond = gr.Radio([(L("None", "Без изменений"), "none")] + [(T.cond[c], c) for c in CONDITIONS],
                                value="glare", label=L("Condition", "Условие"))
                strength = gr.Slider(0.0, 1.0, 0.7, step=0.1, label=L("Strength", "Сила"))
                sim_btn = gr.Button(L("Apply and compare", "Применить и сравнить"), variant="primary")
            with gr.Column(scale=6, min_width=340):
                sim_img = gr.Image(type="pil", label=L("Image the models see", "Что видят модели"), height=230, interactive=False)
                sim_out = gr.HTML()
                gr.Markdown(L("**Where each model looked** (red = most important)", "**Куда смотрела каждая модель** (красное — самое важное)"))
                sim_gal = gr.HTML()
        sim_btn.click(lambda c, k, s: simulate(T, c, k, s), [crop, cond, strength], [sim_img, sim_out, sim_gal])

    with gr.Tab(L("Results and ethics", "Результаты и этика")):
        gr.HTML('<div class="keys">' + "".join(f"<div><b>{v}</b><span>{t}</span></div>" for v, t in key_results) + "</div>")
        gr.Markdown(ETHICS[lang])
        with gr.Accordion(L("All result tables", "Все таблицы результатов"), open=False):
            p = ROOT / "results" / "summary.md"
            gr.Markdown(p.read_text(encoding="utf-8") if p.exists() else "")

    def env_now(request: gr.Request): return env_md(T, request)
    return env, env_now


CSS = """
:root { --sig-red:#E5383B; --sig-yellow:#F2A71B; --sig-green:#19B36B; --housing:#1F242B; --ink:#1C2026; --muted:#5F6773; --line:#DDE2E7; --paper:#FFFFFF; }
.dark { --ink:#E9ECEF; --muted:#A3ABB5; --line:#3A414B; --paper:#1E232A; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; position: relative; }
#lang { position:absolute; right:0; top:10px; z-index:5; width:auto; min-width:0; }
.hero { display:flex; align-items:center; gap:18px; padding: 8px 120px 4px 0; }
.hero h1 { font-size: 30px; line-height: 1.15; margin: 0; letter-spacing: -0.01em; color: var(--ink); }
.hero p { margin: 4px 0 0; color: var(--muted); font-size: 15px; max-width: 62ch; }
.sighead { display:inline-flex; flex-direction:column; gap:6px; padding:8px 7px; background:var(--housing); border-radius:12px; flex:none; }
.sighead .lamp { width:22px; height:22px; border-radius:50%; background:#3A4048; }
.sighead.sm { padding:5px; gap:4px; border-radius:9px; } .sighead.sm .lamp { width:14px; height:14px; }
.lamp.red.on { background:var(--sig-red); box-shadow:0 0 12px 2px rgba(229,56,59,.55); }
.lamp.yellow.on { background:var(--sig-yellow); box-shadow:0 0 12px 2px rgba(242,167,27,.55); }
.lamp.green.on { background:var(--sig-green); box-shadow:0 0 12px 2px rgba(25,179,107,.55); }
.lamp.on.dim { opacity:.55; box-shadow:none; }
.cards { display:flex; flex-direction:column; gap:10px; }
.cards.three { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:10px; }
.card { display:flex; gap:14px; align-items:center; padding:12px 14px; border:1px solid var(--line); border-radius:14px; background:var(--paper); }
.card.unsure { border-style:dashed; }
.card .body { flex:1; min-width:0; }
.card .row { display:flex; align-items:baseline; gap:10px; }
.card .num { color:var(--muted); font-size:14px; font-variant-numeric:tabular-nums; }
.card .title { font-size:22px; font-weight:700; color:var(--ink); }
.card.compact .title { font-size:18px; }
.card .meaning { margin-left:auto; color:var(--muted); font-size:14px; }
.bar { height:6px; background:var(--line); border-radius:3px; margin:8px 0 6px; overflow:hidden; }
.bar .fill { display:block; height:100%; border-radius:3px; }
.fill.red { background:var(--sig-red); } .fill.yellow { background:var(--sig-yellow); } .fill.green { background:var(--sig-green); }
.card .meta { color:var(--muted); font-size:13px; }
.timing { color:var(--muted); font-size:13px; margin-top:10px; }
.empty { padding:18px; border:1px dashed var(--line); border-radius:14px; color:var(--muted); line-height:1.5; }
.empty b { color:var(--ink); font-size:16px; }
.synth { padding:10px 12px; border-left:4px solid var(--sig-yellow); background:rgba(242,167,27,.10); border-radius:6px; color:var(--ink); font-size:14px; margin-bottom:10px; }
.thumbs { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:12px; }
.thumbs figure { margin:0; }
.thumbs img { width:100%; aspect-ratio:1/1; object-fit:contain; background:var(--housing); border-radius:10px; display:block; }
.thumbs figcaption { color:var(--muted); font-size:13px; margin-top:6px; }
.thumbs-empty { color:var(--muted); font-size:14px; }
.keys { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin:6px 0 14px; }
.keys div { padding:14px; border:1px solid var(--line); border-radius:12px; background:var(--paper); }
.keys b { display:block; font-size:22px; color:var(--ink); font-variant-numeric:tabular-nums; }
.keys span { color:var(--muted); font-size:13px; }
.run { min-height:52px; font-size:17px !important; }
footer { display:none !important; }
@media (max-width: 760px) { .cards.three, .keys { grid-template-columns:1fr 1fr; } .hero h1 { font-size:24px; } }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Golos Text"), "Segoe UI", "system-ui", "sans-serif"],
).set(body_background_fill="#F2F4F6", body_background_fill_dark="#15191E",
      block_background_fill="#FFFFFF", block_background_fill_dark="#1E232A",
      block_border_color="#DDE2E7", block_border_color_dark="#3A414B", block_shadow="none", block_label_background_fill="transparent",
      button_primary_background_fill="#1F242B", button_primary_background_fill_hover="#323943",
      button_primary_background_fill_dark="#E9ECEF", button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#15191E",
      button_secondary_background_fill="#FFFFFF", button_secondary_border_color="#C9D0D7",
      checkbox_background_color_selected="#1F242B", color_accent_soft="#E7EBEF", color_accent_soft_dark="#2A3038")

GR6 = int(gr.__version__.split(".")[0]) >= 6
STYLE = dict(css=CSS, theme=THEME)
START = "ru" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1].lower().startswith("ru") else "en"

with gr.Blocks(title="Traffic Light Recognition — A2", **({} if GR6 else STYLE)) as demo:
    lang = gr.Radio([("EN", "en"), ("RU", "ru")], value=START, show_label=False, container=False, elem_id="lang")
    pages, loaders = {}, []
    for code in ["en", "ru"]:
        with gr.Column(visible=code == START) as pages[code]:
            loaders.append(build(code))
    lang.change(lambda v: [gr.update(visible=v == "en"), gr.update(visible=v == "ru")], lang, [pages["en"], pages["ru"]])
    for env, fn in loaders:
        demo.load(fn, None, env)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--lang", default="en", choices=["en", "ru"])
    a = ap.parse_args()
    demo.queue().launch(server_name="127.0.0.1", server_port=a.port, share=a.share, inbrowser=True, **(STYLE if GR6 else {}))
