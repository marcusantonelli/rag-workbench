import { useEffect, useState } from "react";
import { useApp } from "./state";
import { Sidebar } from "./components/Sidebar";
import { Toasts } from "./components/Toasts";
import { Home } from "./views/Home";
import { KbView } from "./views/KbView";
import { SettingsView } from "./views/SettingsView";
import { CommandLineView } from "./views/CommandLineView";
import { onFileDrop } from "./platform";
import { UpdateBanner, useStartupCheck } from "./updates";

export function App() {
  const { screen, kbs, engineError, info, openKb } = useApp();
  const [packHover, setPackHover] = useState(false);
  useStartupCheck();

  // Dropping a .ragpack anywhere opens it (the Build tab handles other files).
  useEffect(
    () =>
      onFileDrop(
        (paths) => paths.filter((p) => p.toLowerCase().endsWith(".ragpack")).forEach((p) => openKb(p)),
        setPackHover,
      ),
    [openKb],
  );

  const kb = screen.kind === "kb" ? kbs.find((k) => k.id === screen.id) : undefined;

  return (
    <div className="app">
      <Sidebar />
      <main className={`main ${packHover ? "drop-hover" : ""}`}>
        <UpdateBanner />
        {!info && (
          <div className="booting">
            <p>{engineError ? "Waiting for the engine…" : "Starting…"}</p>
            {engineError && <p className="muted small">{engineError}</p>}
          </div>
        )}
        {info && screen.kind === "settings" && <SettingsView />}
        {info && screen.kind === "cli" && <CommandLineView />}
        {info && screen.kind === "kb" && kb && <KbView key={kb.id} kb={kb} />}
        {info && (screen.kind === "home" || (screen.kind === "kb" && !kb)) && <Home />}
      </main>
      <Toasts />
    </div>
  );
}
