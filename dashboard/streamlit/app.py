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
        radial-gradient(circle at top left, rgba(48, 97, 90, 0.18), transparent 32rem),
        linear-gradient(135deg, #f7f2e8 0%, #e9f0ea 48%, #dbe7ec 100%);
    }
    .block-container {
      padding-top: 2.3rem;
      max-width: 1180px;
    }
    div[data-testid="stMetric"] {
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(34, 54, 52, 0.12);
      border-radius: 18px;
      padding: 14px 16px;
      box-shadow: 0 14px 30px rgba(30, 45, 48, 0.08);
    }
    .meeting-card {
      padding: 0.2rem 0 0.4rem 0;
    }
    .status-pill {
      display: inline-block;
      border-radius: 999px;
      padding: 0.2rem 0.65rem;
      margin-right: 0.35rem;
      font-size: 0.78rem;
      font-weight: 700;
      background: rgba(32, 70, 64, 0.12);
      color: #1d403b;
    }
    .muted {
      color: rgba(30, 45, 48, 0.72);
      font-size: 0.9rem;
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

    with st.expander(title, expanded=bool(meeting.get("summary_id"))):
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

                rating = st.slider(
                    "Rating",
                    min_value=1,
                    max_value=5,
                    value=int(meeting.get("rating") or 4),
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
