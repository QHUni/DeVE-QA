from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import base64
import io
import json
import os
import re
import time

from .model import CaptionBackend, ReasoningBackend
from .prompts import parse_prompt_json_fallback
from .schema import EventCaption, QAItem, TimeSpan


@dataclass
class OpenAIBackendConfig:
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_sleep: float = 2.0
    temperature: float = 0.0
    max_output_tokens: int = 900
    caption_frame_count: int = 8
    image_max_side: int = 768
    image_quality: int = 85

    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


class OpenAIClientFactory:
    def __init__(self, cfg: OpenAIBackendConfig):
        self.cfg = cfg
        self._client: Any = None

    def get(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - exercised only without dependency
            raise RuntimeError(
                "The OpenAI SDK is not installed. Run `pip install -r requirements.txt` "
                "or `pip install openai`."
            ) from exc
        key = self.cfg.api_key()
        if not key:
            raise RuntimeError(
                f"Missing API key. Export {self.cfg.api_key_env}=your_api_key before using "
                "--llm-backend openai or --caption-backend openai."
            )
        kwargs: Dict[str, Any] = {"api_key": key, "timeout": self.cfg.timeout}
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        self._client = OpenAI(**kwargs)
        return self._client


class OpenAIResponsesMixin:
    def __init__(self, cfg: OpenAIBackendConfig):
        self.openai_cfg = cfg
        self.client_factory = OpenAIClientFactory(cfg)

    def _responses_create(self, *, instructions: str, content: Any) -> str:
        client = self.client_factory.get()
        last_exc: Optional[BaseException] = None
        for attempt in range(max(1, self.openai_cfg.max_retries)):
            try:
                response = client.responses.create(
                    model=self.openai_cfg.model,
                    instructions=instructions,
                    input=[{"role": "user", "content": content}],
                    temperature=self.openai_cfg.temperature,
                    max_output_tokens=self.openai_cfg.max_output_tokens,
                )
                text = getattr(response, "output_text", None)
                if text is not None:
                    return str(text).strip()
                return self._collect_response_text(response)
            except Exception as exc:  
                last_exc = exc
                if attempt + 1 >= max(1, self.openai_cfg.max_retries):
                    break
                time.sleep(self.openai_cfg.retry_sleep * (attempt + 1))
        raise RuntimeError(f"OpenAI Responses API call failed after retries: {last_exc}") from last_exc

    def _collect_response_text(self, response: Any) -> str:
        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {"rationale": cleaned}
        except Exception:
            pass
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {"rationale": cleaned}
            except Exception:
                pass
        return dict(parse_prompt_json_fallback(cleaned))


class OpenAIReasoningBackend(OpenAIResponsesMixin, ReasoningBackend):


    SYSTEM_INSTRUCTIONS = """
You are DeVi, a dense video-event question answering agent. Use only the given
hierarchical event memory and global synopsis. Select exactly one candidate
answer letter and ground the supporting event with the minimum temporal interval
in seconds. Return strict JSON with keys: answer, start, end, confidence,
rationale. Do not include Markdown or extra text.
""".strip()

    def __init__(self, cfg: Optional[OpenAIBackendConfig] = None):
        super().__init__(cfg or OpenAIBackendConfig())

    def answer(self, prompt: str, item: QAItem, captions: Sequence[EventCaption]) -> Mapping[str, object]:
        schema_hint = self._schema_hint(item)
        content = [{"type": "input_text", "text": prompt + "\n\n" + schema_hint}]
        raw_text = self._responses_create(instructions=self.SYSTEM_INSTRUCTIONS, content=content)
        parsed = self._parse_json_object(raw_text)
        normalized = self._normalize_answer(parsed, item, captions, raw_text)
        return normalized

    def _schema_hint(self, item: QAItem) -> str:
        letters = ", ".join(sorted(item.options))
        return (
            "Return only valid JSON, for example: "
            f'{{"answer":"A","start":0.0,"end":10.0,"confidence":0.73,'
            f'"rationale":"..."}}. The answer must be one of: {letters}. '
            "The interval should be the shortest event span that supports the answer."
        )

    def _normalize_answer(self, parsed: Mapping[str, Any], item: QAItem,
                          captions: Sequence[EventCaption], raw_text: str) -> Dict[str, object]:
        options = sorted(item.options) or ["A"]
        ans = str(parsed.get("answer", parsed.get("option", options[0]))).strip().upper()[:1]
        if ans not in item.options:
            ans = self._recover_answer_letter(str(parsed), item) or options[0]
        span = self._recover_span(parsed, captions, item)
        confidence = self._float_in_range(parsed.get("confidence", parsed.get("score", 0.0)))
        rationale = str(parsed.get("rationale", parsed.get("reason", raw_text))).strip()
        return {
            "answer": ans,
            "start": span.start,
            "end": span.end,
            "confidence": confidence,
            "rationale": rationale[:1200],
            "backend": "openai_responses",
            "model": self.openai_cfg.model,
            "raw_text": raw_text[:4000],
        }

    def _recover_answer_letter(self, text: str, item: QAItem) -> Optional[str]:
        for letter in sorted(item.options):
            if re.search(rf"\b{re.escape(letter)}\b", text.upper()):
                return letter
        lowered = text.lower()
        for letter, option in item.options.items():
            if option and option.lower() in lowered:
                return letter
        return None

    def _recover_span(self, parsed: Mapping[str, Any], captions: Sequence[EventCaption], item: QAItem) -> TimeSpan:
        span_value = parsed.get("span", parsed.get("time_span", parsed.get("interval", None)))
        if span_value is not None:
            span = TimeSpan.from_any(span_value)
            if span.duration > 0:
                return span.clamp(item.duration or 10**9)
        start = parsed.get("start", parsed.get("ts", parsed.get("t_start", 0.0)))
        end = parsed.get("end", parsed.get("te", parsed.get("t_end", start)))
        span = TimeSpan(start, end).clamp(item.duration or 10**9)
        if span.duration > 0:
            return span
        if captions:
            return sorted(captions, key=lambda c: c.score, reverse=True)[0].span
        return TimeSpan(0.0, max(1.0, item.duration or 1.0))

    def _float_in_range(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            return 0.0
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))


class OpenAICaptionBackend(OpenAIResponsesMixin, CaptionBackend):

    SYSTEM_INSTRUCTIONS = """
You are a dense-event video captioning model. Given sampled frames from a video
clip and a time interval, describe all visually grounded actions, characters,
objects, interactions, and temporal changes in one concise caption. Avoid
inventing content that is not visible. Return plain text only.
""".strip()

    def __init__(self, cfg: Optional[OpenAIBackendConfig] = None):
        super().__init__(cfg or OpenAIBackendConfig())

    def caption_clip(self, video_path: Optional[str], span: TimeSpan, level: str, prompt: str) -> str:
        if not video_path:
            raise RuntimeError("OpenAI caption backend requires item.video_path or --captions precomputed memory.")
        frames = self._sample_video_frames(video_path, span, self.openai_cfg.caption_frame_count)
        if not frames:
            raise RuntimeError(f"Could not sample frames from video clip: {video_path}")
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": url} for url in frames)
        text = self._responses_create(instructions=self.SYSTEM_INSTRUCTIONS, content=content)
        return self._clean_caption(text, level, span)

    def _sample_video_frames(self, video_path: str, span: TimeSpan, count: int) -> List[str]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video path does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return [self._image_file_to_data_url(path)]
        return self._opencv_sample(path, span, count)

    def _opencv_sample(self, path: Path, span: TimeSpan, count: int) -> List[str]:
        try:
            import cv2
        except Exception as exc:  
            raise RuntimeError(
                "OpenAI video captioning requires opencv-python for frame sampling. "
                "Install optional video packages or pass precomputed --captions."
            ) from exc
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 and total_frames > 0 else span.end
        safe_span = span.clamp(duration)
        if safe_span.duration <= 0:
            safe_span = TimeSpan(0.0, max(1.0, min(duration, span.end or duration or 1.0)))
        times = self._linspace(safe_span.start, safe_span.end, max(1, count))
        urls: List[str] = []
        for sec in times:
            frame_idx = int(max(0, min(total_frames - 1, round(sec * fps)))) if total_frames else int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            urls.append(self._array_to_data_url(frame_rgb))
        cap.release()
        return urls

    def _linspace(self, start: float, end: float, count: int) -> List[float]:
        if count <= 1 or end <= start:
            return [(start + end) / 2.0]
        step = (end - start) / float(count + 1)
        return [start + step * (i + 1) for i in range(count)]

    def _image_file_to_data_url(self, path: Path) -> str:
        try:
            from PIL import Image
        except Exception as exc:  
            raise RuntimeError("Image encoding requires pillow. Install pillow or use precomputed captions.") from exc
        with Image.open(path) as image:
            return self._pil_to_data_url(image.convert("RGB"))

    def _array_to_data_url(self, array: Any) -> str:
        try:
            from PIL import Image
        except Exception as exc:  
            raise RuntimeError("Frame encoding requires pillow. Install pillow or use precomputed captions.") from exc
        image = Image.fromarray(array)
        return self._pil_to_data_url(image)

    def _pil_to_data_url(self, image: Any) -> str:
        max_side = max(64, int(self.openai_cfg.image_max_side))
        width, height = image.size
        scale = min(1.0, max_side / float(max(width, height)))
        if scale < 1.0:
            image = image.resize((int(width * scale), int(height * scale)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(self.openai_cfg.image_quality))
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def _clean_caption(self, text: str, level: str, span: TimeSpan) -> str:
        caption = text.strip().replace("\n", " ")
        caption = re.sub(r"^caption\s*[:\-]\s*", "", caption, flags=re.IGNORECASE)
        if not caption:
            caption = f"A {level} event occurs from {span.start:.1f}s to {span.end:.1f}s."
        return caption[:1200]


def make_openai_backends(model: str = "gpt-4o", use_openai_captioner: bool = False,
                         api_key_env: str = "OPENAI_API_KEY", base_url: Optional[str] = None,
                         max_output_tokens: int = 900, caption_frame_count: int = 8
                         ) -> Tuple[Optional[CaptionBackend], ReasoningBackend]:
    cfg = OpenAIBackendConfig(model=model, api_key_env=api_key_env, base_url=base_url,
                              max_output_tokens=max_output_tokens,
                              caption_frame_count=caption_frame_count)
    reasoning = OpenAIReasoningBackend(cfg)
    captioner: Optional[CaptionBackend] = OpenAICaptionBackend(cfg) if use_openai_captioner else None
    return captioner, reasoning


OPENAI_BACKEND_USAGE = """
export OPENAI_API_KEY="sk-..."
python main.py infer --qa data/sample/qa.jsonl --captions data/sample/captions.jsonl \
  --llm-backend openai --openai-model gpt-4o --output outputs/gpt4o_predictions.jsonl --eval

""".strip()
