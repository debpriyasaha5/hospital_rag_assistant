from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from retrieval.router import route_question
from retrieval.router import clear_chat_history

app = FastAPI(title="Hospital AI Assistant")

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Hospital AI Assistant</title>
    <style>
        :root {
            color-scheme: light;
            --ink: #20332f;
            --muted: #667671;
            --line: #dce6e1;
            --paper: #ffffff;
            --canvas: #f2f6f2;
            --green: #246b58;
            --green-dark: #195342;
            --coral: #c65c43;
            --soft-green: #eaf2ed;
        }
        * { box-sizing: border-box; }
        body {
            min-height: 100vh;
            margin: 0;
            color: var(--ink);
            background-color: var(--canvas);
            background-image: linear-gradient(rgba(36, 107, 88, .035) 1px, transparent 1px), linear-gradient(90deg, rgba(36, 107, 88, .035) 1px, transparent 1px);
            background-size: 28px 28px;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }
        .topbar {
            height: 72px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 max(24px, calc((100vw - 1120px) / 2));
            background: var(--paper);
            border-bottom: 1px solid var(--line);
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-mark {
            width: 36px;
            height: 36px;
            display: grid;
            place-items: center;
            color: white;
            background: var(--green);
            border-radius: 8px;
            font-size: 22px;
            font-weight: 700;
            line-height: 1;
        }
        .brand-name { font-size: 15px; font-weight: 700; }
        .brand-caption { margin-top: 2px; color: var(--muted); font-size: 12px; }
        .service-label {
            color: var(--green-dark);
            background: var(--soft-green);
            padding: 8px 11px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
        }
        main {
            width: min(1120px, calc(100% - 48px));
            margin: 58px auto;
        }
        .page-heading { max-width: 650px; margin-bottom: 30px; }
        .eyebrow {
            margin: 0 0 11px;
            color: var(--coral);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }
        h1 {
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 38px;
            font-weight: 400;
            line-height: 1.15;
        }
        .intro { margin: 12px 0 0; color: var(--muted); font-size: 15px; line-height: 1.6; }
        .workspace {
            display: grid;
            grid-template-columns: minmax(280px, .82fr) minmax(0, 1.18fr);
            border: 1px solid var(--line);
            background: var(--paper);
            box-shadow: 0 14px 40px rgba(32, 51, 47, .07);
        }
        .question-panel { padding: 28px; border-right: 1px solid var(--line); }
        .section-label { margin: 0 0 18px; font-size: 13px; font-weight: 700; }
        label { display: block; margin-bottom: 9px; color: var(--muted); font-size: 12px; font-weight: 700; }
        textarea {
            width: 100%;
            min-height: 132px;
            padding: 14px;
            color: var(--ink);
            background: #fbfcfa;
            border: 1px solid #cbd8d2;
            border-radius: 4px;
            font: inherit;
            font-size: 14px;
            line-height: 1.5;
            resize: vertical;
        }
        textarea:focus { outline: 3px solid rgba(36, 107, 88, .16); border-color: var(--green); }
        button {
            width: 100%;
            min-height: 46px;
            margin-top: 14px;
            color: white;
            background: var(--green);
            border: 0;
            border-radius: 4px;
            font: inherit;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: background-color .16s ease, transform .16s ease;
        }
        button:hover { background: var(--green-dark); }
        button:active { transform: translateY(1px); }
        button:focus-visible { outline: 3px solid rgba(198, 92, 67, .42); outline-offset: 2px; }
        button:disabled { cursor: wait; opacity: .72; }
        .question-note { margin: 13px 0 0; color: var(--muted); font-size: 12px; line-height: 1.5; }
        .answer-panel { min-height: 280px; padding: 28px 32px; }
        .answer-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 17px;
            border-bottom: 1px solid var(--line);
        }
        .answer-heading h2 { margin: 0; font-size: 14px; }
        .route { color: var(--green-dark); font-size: 11px; font-weight: 700; text-transform: uppercase; }
        .answer-content { padding-top: 20px; font-size: 15px; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; }
        .answer-content.empty { color: var(--muted); font-size: 14px; }
        .footer-note { margin: 18px 0 0; color: var(--muted); font-size: 12px; }
        @media (max-width: 720px) {
            .topbar { height: 64px; padding: 0 18px; }
            main { width: min(100% - 32px, 560px); margin: 36px auto; }
            h1 { font-size: 32px; }
            .workspace { grid-template-columns: 1fr; }
            .question-panel { padding: 22px; border-right: 0; border-bottom: 1px solid var(--line); }
            .answer-panel { min-height: 220px; padding: 22px; }
        }
        @media (max-width: 380px) {
            .service-label { padding: 7px; font-size: 10px; }
            .brand-caption { font-size: 11px; }
        }
    </style>
</head>
<body>
    <header class="topbar">
        <div class="brand">
            <div class="brand-mark" aria-hidden="true">+</div>
            <div>
                <div class="brand-name">Sunrise Hospital</div>
                <div class="brand-caption">Patient information desk</div>
            </div>
        </div>
        <div class="service-label">HOSPITAL ASSISTANT</div>
    </header>

    <main>
        <section class="page-heading">
            <p class="eyebrow">Information desk</p>
            <h1>How can we help you today?</h1>
            <p class="intro">Find information about doctors, departments, services, fees, and hospital policies.</p>
        </section>

        <section class="workspace" aria-label="Ask the hospital assistant">
            <div class="question-panel">
                <p class="section-label">Your question</p>
                <form id="askForm">
                    <label for="question">What would you like to know?</label>
                    <textarea id="question" placeholder="For example: Show me the neurologists available." required></textarea>
                    <button id="askButton" type="submit">Ask the assistant</button>
                    <p class="question-note">Answers are based on the hospital information available to this assistant.</p>
                </form>
            </div>
            <section class="answer-panel" aria-live="polite" aria-atomic="true">
                <div class="answer-heading">
                    <h2>Response</h2>
                    <span class="route" id="routeLabel"></span>
                </div>
                <div class="answer-content empty" id="answerContent">Your answer will appear here.</div>
            </section>
        </section>
        <p class="footer-note">For urgent medical needs, contact hospital emergency services directly.</p>
    </main>

    <script>
        const form = document.getElementById('askForm');
        const questionInput = document.getElementById('question');
        const answerContent = document.getElementById('answerContent');
        const routeLabel = document.getElementById('routeLabel');
        const askButton = document.getElementById('askButton');

        form.addEventListener('submit', async function (event) {
            event.preventDefault();
            const question = questionInput.value.trim();
            if (!question) {
                answerContent.textContent = 'Please enter a question.';
                answerContent.classList.add('empty');
                return;
            }

            askButton.disabled = true;
            askButton.textContent = 'Finding an answer...';
            answerContent.textContent = 'Searching hospital information...';
            answerContent.classList.add('empty');
            routeLabel.textContent = '';

            let sessionId = sessionStorage.getItem(
                        "hospitalAssistantSession"
                    )
            
            if(!sessionId){
                sessionId = crypto.randomUUID
                ? crypto.randomUUID()
                : `seesion-${Date.now()}-${Math.random()}`;
    
                sessionStorage.setItem(
                "hospitalAssistantSession",
                sessionId
                )
            }

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, session_id: sessionId })
                });

                if (!response.ok) {
                    throw new Error('Request failed');
                }

                const data = await response.json();
                answerContent.textContent = data.answer || 'No answer returned.';
                answerContent.classList.remove('empty');
                routeLabel.textContent = data.route ? `Route: ${data.route}` : '';
            } catch (error) {
                answerContent.textContent = 'Something went wrong while fetching the answer. Please try again.';
                answerContent.classList.add('empty');
            } finally {
                askButton.disabled = false;
                askButton.textContent = 'Ask the assistant';
            }
        });
    </script>
</body>
</html>
"""


class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    route: str
    answer: str
    contexts: list[str] = Field(default_factory=list)


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_hospital_question(payload: QuestionRequest) -> AskResponse:
    result = route_question(payload.question, session_id=payload.session_id)
    return AskResponse(**result)

@app.delete("/session/{session_id}")
def clear_hospital_session(session_id: str):
    clear_chat_history(session_id)

    return {
        "status": "cleared"
    }

