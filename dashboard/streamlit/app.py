import os
from datetime import datetime
from typing import Any

import requests
import streamlit as st


DATA_API_URL = os.environ.get(
    "DATA_API_URL",
    "http://data-api.platform.svc.cluster.local:8000",
).rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))


st.set_page_config(
    page_title="Meeting Intelligence",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
      background:
        radial-gradient(circle at top left, rgba(30, 115, 109, 0.34), transparent 30rem),
        radial-gradient(circle at bottom right, rgba(222, 165, 86, 0.2), transparent 34rem),
        linear-gradient(135deg, #061014 0%, #0b1820 48%, #12161f 100%);
      color: #f4fbf7;
    }
    .block-container {
      padding-top: 2.3rem;
      max-width: 1180px;
    }
    h1, h2, h3, h4, h5, h6,
    p, span, label, div[data-testid="stMarkdownContainer"] {
      color: #f4fbf7 !important;
    }
    .stCaptionContainer, .stCaptionContainer p {
      color: rgba(244, 251, 247, 0.72) !important;
    }
    div[data-testid="stMetric"] {
      background: linear-gradient(145deg, rgba(19, 32, 41, 0.94), rgba(14, 24, 31, 0.86));
      border: 1px solid rgba(255, 255, 255, 0.13);
      border-radius: 18px;
      padding: 14px 16px;
      box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"] {
      color: #ffffff !important;
    }
    .meeting-card {
      padding: 0.2rem 0 0.4rem 0;
    }
    div[data-testid="stExpander"] {
      background: rgba(12, 20, 27, 0.82);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 18px;
      margin-bottom: 14px;
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
      overflow: hidden;
    }
    div[data-testid="stExpander"] summary {
      min-height: 3.25rem;
      padding: 0.6rem 1rem;
      color: #ffffff !important;
      font-weight: 800;
      letter-spacing: 0.01em;
    }
    div[data-testid="stExpander"] summary p {
      color: #ffffff !important;
      font-size: 1.02rem;
    }
    .status-pill {
      display: inline-block;
      border-radius: 999px;
      padding: 0.2rem 0.65rem;
      margin-right: 0.35rem;
      font-size: 0.78rem;
      font-weight: 700;
      background: rgba(78, 221, 177, 0.14);
      color: #9ff6d7 !important;
      border: 1px solid rgba(159, 246, 215, 0.18);
    }
    .muted {
      color: rgba(244, 251, 247, 0.7) !important;
      font-size: 0.9rem;
    }
    code {
      color: #68f0bd !important;
      background: rgba(104, 240, 189, 0.12) !important;
      border-radius: 7px;
      padding: 0.1rem 0.35rem;
    }
    .stButton button, .stFormSubmitButton button {
      background: linear-gradient(135deg, #f0b35f, #f5695f);
      color: #101820 !important;
      border: 0;
      border-radius: 12px;
      font-weight: 800;
      box-shadow: 0 12px 26px rgba(245, 105, 95, 0.22);
    }
    textarea, input {
      background-color: rgba(8, 14, 20, 0.74) !important;
      color: #ffffff !important;
      border-color: rgba(255, 255, 255, 0.16) !important;
    }
    div[role="radiogroup"] label p {
      color: #ffd166 !important;
      font-weight: 900 !important;
      letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    resp = requests.request(
        method,
        f"{DATA_API_URL}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


@st.cache_data(ttl=8)
def fetch_meetings(limit: int) -> list[dict[str, Any]]:
    resp = _request(
        "GET",
        "/meetings",
        params={"limit": limit, "source_prefix": "jitsi_recording:"},
    )
    return resp.json()


def submit_review(
    meeting_id: str,
    rating: int,
    approved: bool,
    edited_summary: str,
    edited_action_items: str,
) -> str:
    resp = _request(
        "POST",
        "/reviews",
        json={
            "meeting_id": meeting_id,
            "rating": rating,
            "approved": approved,
            "edited_summary": edited_summary,
            "edited_action_items": edited_action_items,
        },
    )
    return resp.json()["review_id"]


def _short_id(value: str | None) -> str:
    return value[:8] if value else "unknown"


def _format_time(value: str | None) -> str:
    if not value:
        return "unknown time"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%b %d, %H:%M")
    except ValueError:
        return value


def _preview(text: str | None, limit: int = 420) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit].rstrip()}..."


def _stars(score: int) -> str:
    return "★" * score + "☆" * (5 - score)


st.title("Meeting Intelligence Dashboard")
st.caption("Live summaries, transcripts, and feedback for Jitsi recordings")

top_left, top_right = st.columns([3, 1])
with top_left:
    limit = st.slider("Meetings to show", min_value=5, max_value=100, value=25, step=5)
with top_right:
    st.write("")
    st.write("")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    meetings = fetch_meetings(limit)
except Exception as exc:
    st.error(f"Could not load meetings from data-api: {exc}")
    st.stop()

total = len(meetings)
transcribed = sum(1 for m in meetings if m.get("transcript_id"))
summarized = sum(1 for m in meetings if m.get("summary_id"))
reviewed = sum(1 for m in meetings if m.get("review_id"))

metric_cols = st.columns(4)
metric_cols[0].metric("Meetings", total)
metric_cols[1].metric("Transcribed", transcribed)
metric_cols[2].metric("Summarized", summarized)
metric_cols[3].metric("Reviewed", reviewed)

st.divider()

if not meetings:
    st.info("No Jitsi recordings have been captured yet.")
    st.stop()

for meeting in meetings:
    meeting_id = meeting["meeting_id"]
    created_at = _format_time(meeting.get("created_at"))
    title = f"{created_at} · {_short_id(meeting_id)}"
    if meeting.get("summary_id"):
        title += " · summary ready"
    elif meeting.get("transcript_id"):
        title += " · transcript ready"
    else:
        title += f" · ASR {meeting.get('asr_status', 'unknown')}"

    with st.expander(title, expanded=False):
        st.markdown("<div class='meeting-card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <span class="status-pill">meeting: {meeting.get("status")}</span>
            <span class="status-pill">asr: {meeting.get("asr_status")}</span>
            <span class="status-pill">source: {meeting.get("source", "").split(":")[0]}</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='muted'>meeting_id: <code>{meeting_id}</code><br>"
            f"audio: <code>{meeting.get('audio_object_key')}</code></p>",
            unsafe_allow_html=True,
        )

        if meeting.get("asr_last_error"):
            st.warning(f"ASR error: {meeting['asr_last_error']}")

        transcript_col, summary_col = st.columns([1, 1])
        with transcript_col:
            st.subheader("Transcript")
            if meeting.get("transcript_text"):
                st.write(_preview(meeting["transcript_text"], limit=900))
            else:
                st.info("Transcript is not ready yet.")

        with summary_col:
            st.subheader("Summary")
            if meeting.get("summary_text"):
                st.write(meeting["summary_text"])
                action_items = meeting.get("action_item_text") or ""
                if action_items.strip():
                    st.subheader("Action Items")
                    st.write(action_items)
            else:
                st.info("Summary is not ready yet.")

        if meeting.get("summary_text"):
            st.subheader("Feedback")
            if meeting.get("review_id"):
                st.caption(
                    f"Latest review: rating {meeting.get('rating')} / 5, "
                    f"approved={meeting.get('approved')}"
                )

            with st.form(f"feedback-{meeting_id}"):
                default_summary = meeting.get("edited_summary") or meeting.get("summary_text") or ""
                default_actions = meeting.get("edited_action_items")
                if default_actions is None:
                    default_actions = meeting.get("action_item_text") or ""

                st.markdown("**Rating**")
                rating = st.radio(
                    "Rating",
                    options=[1, 2, 3, 4, 5],
                    index=max(0, min(4, int(meeting.get("rating") or 4) - 1)),
                    format_func=_stars,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"rating-{meeting_id}",
                )
                approved = st.checkbox(
                    "Approve this summary for retraining",
                    value=bool(meeting.get("approved") if meeting.get("approved") is not None else True),
                    key=f"approved-{meeting_id}",
                )
                edited_summary = st.text_area(
                    "Edited summary",
                    value=default_summary,
                    height=140,
                    key=f"summary-{meeting_id}",
                )
                edited_action_items = st.text_area(
                    "Edited action items",
                    value=default_actions,
                    height=90,
                    key=f"actions-{meeting_id}",
                )
                submitted = st.form_submit_button("Submit feedback", use_container_width=True)

            if submitted:
                try:
                    review_id = submit_review(
                        meeting_id,
                        rating,
                        approved,
                        edited_summary,
                        edited_action_items,
                    )
                    st.success(f"Feedback saved: {review_id}")
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Could not save feedback: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)
