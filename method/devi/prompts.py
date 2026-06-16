from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .schema import EventCaption, QAItem, Prediction


CAPTION_PROMPT = """
You are a dense-event video captioner. Given sampled frames from a {level}
video clip spanning [{start:.1f}s, {end:.1f}s], describe every visible human
activity, object interaction, and temporal change. Use one concise sentence.
Avoid guessing identities, avoid hallucinating unseen objects, and include the
purpose of the action only when it is visually supported.
""".strip()

CONTEXTUALIZATION_PROMPT = """
You are a highly intelligent language agent improving video captions. Given a
set of captions, each representing a different time segment and hierarchy level,
and a question about the video, refine each caption by incorporating contextual
information from all the other captions and the question. Analyze the overall
narrative, identify relevant context, and preserve temporal coherence.

Question:
{question}

Hierarchical captions:
{captions}

Return JSON with two fields:
1. refined_captions: a list with the same number of captions, each containing
   text, start, end, and level.
2. synopsis: a comprehensive synopsis of the entire video covering key temporal
actions, characters, and interactions.
""".strip()

QA_PROMPT = """
You are a helpful expert in dense-event video question answering. Select the
correct answer from the candidate answer set based on the event descriptions.
Also output the minimum time interval [t_s, t_e] of the event that supports the
answer. The selected interval should be precise and should overlap the event
where the answer is visually grounded.

Event descriptions:
{event_memory}

Question:
{question}

Candidate answers:
{options}

Return JSON exactly in this schema:
{{"answer": "A", "start": 0.0, "end": 1.0, "rationale": "..."}}
""".strip()

VERIFY_PROMPT = """
You are a helpful expert in dense event video analysis. You previously answered
a multiple-choice question and gave the minimum time interval for support.
However, a professional consistency check found that the similarity between the
previous answer "{previous_answer}" and the supportive frames/captions was only
{score:.3f}, below the threshold {threshold:.3f}.

Previous prediction:
{prediction}

Please answer again using the event descriptions and candidate answers. If your
answer changes, analyze the inconsistency. If it stays the same, explain how the
answer relates to the video evidence.

{qa_prompt}
""".strip()

TRAIN_PROMPT = """
Training objective for the lightweight DeVi ranker:
Given a question, answer candidates, and event captions, learn a text scoring
function that ranks the gold answer and gold event span above distractors. This
is not intended to replace an MLLM; it provides a runnable local training path
and a useful fallback when API access is unavailable.
""".strip()


@dataclass
class PromptPack:
    caption_prompt: str = CAPTION_PROMPT
    contextualization_prompt: str = CONTEXTUALIZATION_PROMPT
    qa_prompt: str = QA_PROMPT
    verify_prompt: str = VERIFY_PROMPT
    train_prompt: str = TRAIN_PROMPT

    def as_dict(self) -> Dict[str, str]:
        return {
            "caption_prompt": self.caption_prompt,
            "contextualization_prompt": self.contextualization_prompt,
            "qa_prompt": self.qa_prompt,
            "verify_prompt": self.verify_prompt,
            "train_prompt": self.train_prompt,
        }


def format_captions(captions: Sequence[EventCaption], max_chars: int = 12000) -> str:
    lines: List[str] = []
    total = 0
    for cap in sorted(captions, key=lambda c: (c.span.start, c.span.end, c.level)):
        line = cap.to_text_block()
        if total + len(line) > max_chars:
            lines.append("... [caption memory truncated for context length]")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines) if lines else "No captions are available."


def format_options(item: QAItem) -> str:
    return item.option_block()


def build_caption_prompt(level: str, start: float, end: float, pack: PromptPack | None = None) -> str:
    pack = pack or PromptPack()
    return pack.caption_prompt.format(level=level, start=start, end=end)


def build_contextualization_prompt(item: QAItem, captions: Sequence[EventCaption], pack: PromptPack | None = None) -> str:
    pack = pack or PromptPack()
    return pack.contextualization_prompt.format(question=item.question, captions=format_captions(captions))


def build_qa_prompt(item: QAItem, captions: Sequence[EventCaption], synopsis: str = "", pack: PromptPack | None = None) -> str:
    pack = pack or PromptPack()
    memory = format_captions(captions)
    if synopsis:
        memory += "\nGlobal synopsis: " + synopsis
    return pack.qa_prompt.format(event_memory=memory, question=item.question, options=format_options(item))


def build_verify_prompt(item: QAItem, captions: Sequence[EventCaption], prediction: Prediction,
                        previous_answer_text: str, threshold: float, pack: PromptPack | None = None) -> str:
    pack = pack or PromptPack()
    qa_prompt = build_qa_prompt(item, captions, prediction.raw.get("synopsis", ""), pack)
    return pack.verify_prompt.format(previous_answer=previous_answer_text, score=prediction.consistency,
                                     threshold=threshold, prediction=prediction.to_dict(), qa_prompt=qa_prompt)


def parse_prompt_json_fallback(text: str) -> Dict[str, object]:
    import json
    import re

    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    ans = re.search(r"answer\W*([A-E])", text, flags=re.IGNORECASE)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    return {"answer": ans.group(1).upper() if ans else "A", "start": nums[0] if nums else 0.0,
            "end": nums[1] if len(nums) > 1 else (nums[0] if nums else 0.0), "rationale": text[:500]}


def few_shot_examples() -> str:
    examples = [
        ("A person is holding a knife and waving it around.", "A person is holding a knife and chopping down a tree."),
        ("A person jumps into the water.", "A person jumps into the water to save someone who is drowning."),
        ("A man is holding scissors near a cat.", "An old man is cutting the black cat's claws with scissors."),
    ]
    return "\n".join(f"Original Caption: {a}\nContextualized Caption: {b}" for a, b in examples)


def project_readme_text() -> str:
    return """
DeVi Dense Video Events implements hierarchical captioning, temporal event
memory, event-grounded QA, and self-consistency verification.  It can run in a
local lexical mock mode for debugging, or you can replace the backend classes
with LLaMA-VID/GPT/Gemini/CLIP calls for full experiments.
""".strip()


class PromptDebugger:

    def __init__(self, pack: PromptPack | None = None):
        self.pack = pack or PromptPack()

    def preview(self, item: QAItem) -> Dict[str, str]:
        caps = item.captions
        first = caps[0] if caps else EventCaption(item.video_id, "No event", item.gt_span or [0, 0])
        return {
            "caption": build_caption_prompt(first.level, first.span.start, first.span.end, self.pack),
            "contextualization": build_contextualization_prompt(item, caps, self.pack),
            "qa": build_qa_prompt(item, caps, "", self.pack),
        }
