import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from retrieval.router import clear_chat_history, route_question
from utils.logging_config import configure_logging


logger = configure_logging(__name__)

st.set_page_config(
	page_title="Sunrise Hospital Assistant",
	page_icon="+",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
		:root {
			color-scheme: light;
			--hospital-ink: #20332f;
			--hospital-muted: #667671;
			--hospital-line: #dce6e1;
			--hospital-paper: #ffffff;
			--hospital-canvas: #f2f6f2;
			--hospital-green: #246b58;
			--hospital-green-dark: #195342;
			--hospital-coral: #c65c43;
			--hospital-soft-green: #eaf2ed;
		}
		* { box-sizing: border-box; }
		[data-testid="stAppViewContainer"] {
			min-height: 100vh;
			color: var(--hospital-ink);
			background-color: var(--hospital-canvas);
			background-image:
				linear-gradient(rgba(36, 107, 88, .035) 1px, transparent 1px),
				linear-gradient(90deg, rgba(36, 107, 88, .035) 1px, transparent 1px);
			background-size: 28px 28px;
		}
		[data-testid="stHeader"] { background: rgba(242, 246, 242, .88); }
		.block-container { max-width: 1120px; padding-top: 4rem; padding-bottom: 7rem; }
		[data-testid="stSidebar"] {
			background: var(--hospital-paper);
			border-right: 1px solid var(--hospital-line);
		}
		.brand-row {
			display: flex; align-items: center; gap: 12px;
			flex: 1 1 220px; min-width: 0; margin: 0 0 2.4rem;
		}
		.brand-mark {
			display: grid; place-items: center; width: 38px; height: 38px;
			border-radius: 8px; color: white; background: var(--hospital-green);
			flex: 0 0 38px; font: 700 23px/1 "Trebuchet MS", sans-serif;
		}
		.brand-name {
			color: var(--hospital-ink); font-size: 15px; font-weight: 700;
			overflow-wrap: anywhere;
		}
		.brand-caption {
			margin-top: 2px; color: var(--hospital-muted); font-size: 12px;
			overflow-wrap: anywhere;
		}
		.service-label {
			display: inline-block; padding: 8px 11px; border-radius: 4px;
			color: var(--hospital-green-dark); background: var(--hospital-soft-green);
			font-size: 12px; font-weight: 700;
		}
		.topbar {
			display: flex; flex-wrap: wrap; align-items: center;
			justify-content: space-between; gap: 12px;
			margin: 0 0 2.5rem; padding: 0 0 1rem;
			border-bottom: 1px solid var(--hospital-line);
		}
		.eyebrow {
			margin: 0 0 .65rem; color: var(--hospital-coral); font-size: 12px;
			font-weight: 700; text-transform: uppercase;
		}
		h1, h2, h3 { color: var(--hospital-ink); }
		h1 { font-family: Georgia, "Times New Roman", serif; font-size: 38px; font-weight: 400; line-height: 1.15; }
		.page-intro { max-width: 650px; margin: .75rem 0 2rem; color: var(--hospital-muted); font-size: 15px; line-height: 1.6; }
		[data-testid="stChatMessage"] {
			border: 1px solid var(--hospital-line);
			border-radius: 8px;
			background: rgba(255, 255, 255, .92);
		}
		[data-testid="stChatInput"] { border-color: #cbd8d2; }
		[data-testid="stChatInput"] textarea { color: var(--hospital-ink); }
		[data-testid="stChatInput"]:focus-within { border-color: var(--hospital-green); }
		[data-testid="stButton"] button {
			border-radius: 5px;
			border-color: var(--hospital-line);
			color: var(--hospital-ink);
		}
		[data-testid="stButton"] button:hover {
			color: var(--hospital-green-dark);
			border-color: var(--hospital-green);
		}
		.footer-note { color: var(--hospital-muted); font-size: 12px; }
		@media (max-width: 700px) {
			.block-container { padding: 4rem 1rem 6rem; }
			.topbar { align-items: flex-start; margin-bottom: 2rem; }
			.service-label { margin-left: 50px; }
		}
	</style>
	""",
	unsafe_allow_html=True,
)

st.session_state.setdefault("hospital_session_id", uuid4().hex)
st.session_state.setdefault("chat_messages", [])

with st.sidebar:
	st.subheader("Conversation")
	if st.button("Clear conversation", use_container_width=True):
		clear_chat_history(st.session_state.hospital_session_id)
		st.session_state.chat_messages = []
		st.rerun()
	st.divider()
	st.markdown(
		'<p class="footer-note">For urgent medical needs, contact hospital emergency services directly.</p>',
		unsafe_allow_html=True,
	)

st.markdown(
	"""
	<div class="topbar">
		<div class="brand-row" style="margin: 0">
			<div class="brand-mark">+</div>
			<div>
				<div class="brand-name">Sunrise Hospital</div>
				<div class="brand-caption">Patient information desk</div>
			</div>
		</div>
		<div class="service-label">HOSPITAL ASSISTANT</div>
	</div>
	<p class="eyebrow">Information desk</p>
	""",
	unsafe_allow_html=True,
)
st.title("How can we help you today?")
st.markdown(
	'<p class="page-intro">Find information about doctors, departments, services, fees, and hospital policies.</p>',
	unsafe_allow_html=True,
)

for message in st.session_state.chat_messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])
		if message.get("route"):
			st.caption(f"Retrieved via {message['route']} search")

question = st.chat_input("Ask about doctors, services, fees, or hospital policies")

if question:
	st.session_state.chat_messages.append({"role": "user", "content": question})
	with st.chat_message("user"):
		st.markdown(question)

	with st.chat_message("assistant"):
		with st.spinner("Searching hospital information..."):
			try:
				result = route_question(
					question,
					session_id=st.session_state.hospital_session_id,
				)
				answer = result.get("answer", "No answer returned.")
				route = result.get("route")
			except Exception:
				logger.exception("Streamlit question request failed")
				answer = "Something went wrong while finding an answer. Please try again."
				route = None

		st.markdown(answer)
		if route:
			st.caption(f"Retrieved via {route} search")

	st.session_state.chat_messages.append(
		{"role": "assistant", "content": answer, "route": route}
	)