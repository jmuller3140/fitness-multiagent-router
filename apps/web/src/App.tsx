import { Activity, Dumbbell, LoaderCircle, Route, Send, TerminalSquare } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";

type RouteName = "COACH" | "WORKOUT_GENERATE" | "WORKOUT_LOG" | "FALLBACK";

type ChatResponse = {
  session_id: string;
  route: RouteName;
  confidence: number;
  reason: string;
  final_response: string;
  structured_output: Record<string, unknown> | null;
  errors: string[];
};

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const EXAMPLES = [
  "What muscles does a deadlift work?",
  "Build me a 30 min upper body session with dumbbells",
  "I just did 3x10 bench press at 185 lbs",
  "Bench press",
];

const routeLabels: Record<RouteName, string> = {
  COACH: "Coach",
  WORKOUT_GENERATE: "Generate",
  WORKOUT_LOG: "Log",
  FALLBACK: "Clarify",
};

export function App() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [input, setInput] = useState(EXAMPLES[1]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const latestRoute = useMemo(() => {
    const latest = [...messages].reverse().find((message) => message.response);
    return latest?.response?.route ?? "COACH";
  }, [messages]);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const text = input.trim();
    if (!text || isSending) return;

    setInput("");
    setError(null);
    setMessages((current) => [...current, { role: "user", text }]);
    setIsSending(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const data = (await response.json()) as ChatResponse;
      setMessages((current) => [
        ...current,
        { role: "assistant", text: data.final_response, response: data },
      ]);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Unknown request failure";
      setError(message);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: "The API is unavailable. Start the Python service and try again.",
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="shell">
      <section className="workspace">
        <aside className="sidebar" aria-label="Router state">
          <div className="brand">
            <Dumbbell size={26} aria-hidden />
            <div>
              <h1>Fitness Router</h1>
              <p>LangGraph hub + DSPy intent routing</p>
            </div>
          </div>

          <div className="routePanel">
            <div className="panelTitle">
              <Route size={17} aria-hidden />
              <span>Selected Route</span>
            </div>
            <strong>{routeLabels[latestRoute]}</strong>
          </div>

          <div className="examples">
            <div className="panelTitle">
              <TerminalSquare size={17} aria-hidden />
              <span>Examples</span>
            </div>
            {EXAMPLES.map((example) => (
              <button key={example} type="button" onClick={() => setInput(example)}>
                {example}
              </button>
            ))}
          </div>
        </aside>

        <section className="chatSurface" aria-label="Fitness agent chat">
          <div className="messageList">
            {messages.length === 0 ? (
              <div className="emptyState">
                <Activity size={38} aria-hidden />
                <h2>Send a coaching, workout generation, or workout log request.</h2>
              </div>
            ) : (
              messages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                  <p>{message.text}</p>
                  {message.response ? <RouteDebug response={message.response} /> : null}
                </article>
              ))
            )}
          </div>

          {error ? <div className="errorBanner">API error: {error}</div> : null}

          <form className="composer" onSubmit={submit}>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask, generate, or log a workout..."
              aria-label="Message"
            />
            <button type="submit" disabled={isSending} aria-label="Send message">
              {isSending ? <LoaderCircle className="spin" size={20} /> : <Send size={20} />}
            </button>
          </form>
        </section>
      </section>
    </main>
  );
}

function RouteDebug({ response }: { response: ChatResponse }) {
  return (
    <details className="debug">
      <summary>
        {routeLabels[response.route]} / {(response.confidence * 100).toFixed(0)}%
      </summary>
      <dl>
        <dt>Reason</dt>
        <dd>{response.reason}</dd>
      </dl>
      {response.structured_output ? (
        <pre>{JSON.stringify(response.structured_output, null, 2)}</pre>
      ) : null}
    </details>
  );
}
