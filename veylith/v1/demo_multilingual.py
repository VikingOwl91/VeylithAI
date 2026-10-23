#!/usr/bin/env python3
"""Demo: how multilingual tokenizers work, measured on a toy corpus.

Trains a tiny multilingual BPE tokenizer (vocab 800) on a hand-written
sample of 11 languages, then compares token fertility against the
English-only Shakespeare tokenizer from train.py:

    fertility = tokens per character (all languages)
              = tokens per word (space-separated languages)

Artifacts go to /tmp/opencode/multilingual_demo; only this script lives
in the repo.

Run:  python demo_multilingual.py
"""

import os
import unicodedata
from collections import Counter

import sentencepiece as spm

OUT_DIR = "/tmp/opencode/multilingual_demo"
ENGLISH_MODEL = "data/tinyshakespeare_bpe1000.model"
VOCAB_SIZE = 800

SAMPLES = {
    "en": ["The quick brown fox jumps over the lazy dog.",
           "Machine learning is changing the world we live in.",
           "Artificial intelligence is becoming more important every day."],
    "de": ["Der schnelle braune Fuchs springt über den faulen Hund.",
           "Maschinelles Lernen verändert die Welt, in der wir leben.",
           "Künstliche Intelligenz wird immer wichtiger."],
    "fr": ["Le chat noir dort sur le tapis rouge.",
           "L'apprentissage automatique change le monde dans lequel nous vivons.",
           "L'intelligence artificielle devient de plus en plus importante."],
    "es": ["El perro corre rápido por el parque.",
           "El aprendizaje automático está cambiando el mundo.",
           "La inteligencia artificial es cada vez más importante."],
    "zh": ["机器学习正在改变我们生活的世界。",
           "人工智能是未来最关键的技术之一。",
           "科学技术每天都在快速发展。"],
    "ja": ["桜の花が春に咲きます。",
           "機械学習は私たちの生活を変えています。",
           "東京は日本の首都です。"],
    "ru": ["Быстрая коричневая лиса прыгает через ленивую собаку.",
           "Машинное обучение меняет мир, в котором мы живем.",
           "Технологии развиваются очень быстро."],
    "ar": ["الثعلب البني السريع يقفز فوق الكلب الكسول.",
           "التعلم الآلي يغير العالم من حولنا.",
           "التكنولوجيا تتطور بسرعة كل يوم."],
    "hi": ["मशीन लर्निंग दुनिया बदल रही है।",
           "तकनीक हर दिन तेज़ी से आगे बढ़ रही है।",
           "भारत में तकनीकी विकास बहुत तेज़ है।"],
    "ko": ["기계 학습이 세상을 바꾸고 있습니다.",
           "기술은 매일 빠르게 발전하고 있습니다.",
           "인공지능은 미래의 핵심 기술입니다."],
    "th": ["การเรียนรู้ของเครื่องกำลังเปลี่ยนโลก",
           "เทคโนโลยีพัฒนาอย่างรวดเร็วทุกวัน",
           "ปัญญาประดิษฐ์เป็นเทคโนโลยีแห่งอนาคต"],
}


def script_of(ch):
    name = unicodedata.name(ch, "")
    if "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name:
        return "han"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "kana"
    if "HANGUL" in name:
        return "hangul"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "ARABIC" in name:
        return "arabic"
    if "DEVANAGARI" in name:
        return "devanagari"
    if "THAI" in name:
        return "thai"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return "other"


def piece_script(piece):
    if piece.startswith("<0x") and piece.endswith(">"):
        return "byte-fallback"
    if piece.startswith("<") and piece.endswith(">"):
        return "special"
    scripts = {script_of(c) for c in piece if c.isalpha()}
    if len(scripts) == 1:
        return scripts.pop()
    return "mixed" if scripts else "digits/punct"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus_path = os.path.join(OUT_DIR, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for sentences in SAMPLES.values():
            for s in sentences:
                f.write(s + "\n")
    n_chars = sum(len(s) for ss in SAMPLES.values() for s in ss)
    n_sent = sum(len(ss) for ss in SAMPLES.values())
    print(f"corpus: {n_chars:,} chars, {n_sent} sentences, {len(SAMPLES)} languages\n")

    model_prefix = os.path.join(OUT_DIR, f"multi_bpe{VOCAB_SIZE}")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        byte_fallback=True,
        normalization_rule_name="identity",
        character_coverage=1.0,
        hard_vocab_limit=False,
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
    )
    multi_path = model_prefix + ".model"

    en = spm.SentencePieceProcessor(model_file=ENGLISH_MODEL)
    multi = spm.SentencePieceProcessor(model_file=multi_path)
    print(f"tokenizers: english-only vocab {en.get_piece_size():,} | "
          f"multilingual vocab {multi.get_piece_size():,}\n")

    print(f"{'lang':5s} {'chars':>6s} {'en-only tok/char':>17s} {'multi tok/char':>15s} {'multi tok/word':>15s}")
    for lang, sentences in SAMPLES.items():
        text = " ".join(sentences)
        n_en = len(en.encode(text, out_type=int))
        n_multi = len(multi.encode(text, out_type=int))
        n_words = len(text.split())
        print(f"{lang:5s} {len(text):6d} {n_en / len(text):17.2f} "
              f"{n_multi / len(text):15.2f} {n_multi / n_words:15.2f}")

    zh_text = " ".join(SAMPLES["zh"])
    han = sum(1 for c in zh_text if script_of(c) == "han")
    print(f"\nChinese: {han} Han characters -> "
          f"{len(en.encode(zh_text, out_type=int)) / han:.2f} tok/char (english-only) vs "
          f"{len(multi.encode(zh_text, out_type=int)) / han:.2f} tok/char (multilingual)")

    print("\nvocab composition of the multilingual tokenizer:")
    counts = Counter()
    with open(model_prefix + ".vocab", encoding="utf-8") as f:
        for line in f:
            counts[piece_script(line.split("\t")[0])] += 1
    total = sum(counts.values())
    for k, v in counts.most_common():
        print(f"  {k:15s} {v:5d} ({v / total:.1%})")

    print("\nlearned pieces per language (multilingual tokenizer):")
    probes = [("de", "Maschinelles Lernen verändert die Welt"),
              ("zh", "机器学习正在改变世界"),
              ("ja", "機械学習は生活を変えています"),
              ("ru", "Машинное обучение меняет мир"),
              ("ar", "التعلم الآلي يغير العالم"),
              ("ko", "기계 학습이 세상을 바꾸고 있습니다"),
              ("hi", "मशीन लर्निंग दुनिया बदल रही है"),
              ("th", "การเรียนรู้ของเครื่องกำลังเปลี่ยนโลก")]
    for lang, probe in probes:
        print(f"  {lang}: {multi.encode(probe, out_type=str)}")

    vi = "Học máy đang thay đổi thế giới."  # Vietnamese: not in the training sample
    ids = multi.encode(vi, out_type=int)
    print(f"\nunseen language (Vietnamese, not in training data): "
          f"{len(ids)} tokens for {len(vi)} chars, roundtrip exact: {multi.decode(ids) == vi}")


if __name__ == "__main__":
    main()
