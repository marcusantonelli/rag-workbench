import { useApp } from "../state";
import { call } from "../api";

/** The embedding model's state, always visible; click to manage it. */
export function RuntimeBadge() {
  const { runtime, setScreen, fail, refreshRuntime } = useApp();
  if (!runtime) return null;

  let tone = "off";
  let title = "Embedding model stopped";
  let sub = "Click to start";
  if (runtime.mode === "external") {
    tone = "ready";
    title = "Using your llama-server";
    sub = runtime.external_url;
  } else if (runtime.needs_setup) {
    tone = "warn";
    title = "Embedding model not set up";
    sub = "Needed to search and build";
  } else if (runtime.state === "ready") {
    tone = "ready";
    title = "Embedding model ready";
    const gpu = runtime.device_note.startsWith("Running on") ? runtime.device_note.replace(/^Running on /, "").replace(/\.$/, "") : "CPU";
    sub = gpu;
  } else if (runtime.state === "starting") {
    tone = "busy";
    title = "Loading the embedding model";
    sub = "This takes up to a minute";
  } else if (runtime.state === "failed") {
    tone = "error";
    title = "Embedding model failed";
    sub = "Open Settings for the log";
  }

  const onClick = async () => {
    if (!runtime.needs_setup && runtime.mode === "managed" && runtime.state === "stopped") {
      try {
        await call("runtime.start");
        refreshRuntime();
      } catch (e) {
        fail(e, "Couldn't start the embedding model");
      }
      return;
    }
    setScreen({ kind: "settings" });
  };

  return (
    <button className={`runtime-badge tone-${tone}`} onClick={onClick} title={runtime.error ?? runtime.device_note}>
      <span className="runtime-dot" aria-hidden="true" />
      <span className="runtime-text">
        <span className="runtime-title">{title}</span>
        <span className="runtime-sub">{sub}</span>
      </span>
    </button>
  );
}
