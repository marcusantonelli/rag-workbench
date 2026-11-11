import { useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronDown, X } from "lucide-react";

export function Button({
  children, onClick, kind = "secondary", disabled, title, type = "button", icon,
}: {
  children?: ReactNode;
  onClick?: () => void;
  kind?: "primary" | "secondary" | "ghost" | "danger";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
  icon?: ReactNode;
}) {
  return (
    <button type={type} className={`btn btn-${kind}`} onClick={onClick} disabled={disabled} title={title}>
      {icon}
      {children && <span>{children}</span>}
    </button>
  );
}

export interface MenuItem {
  label: string;
  /** A second, quieter line saying what the choice does. */
  detail?: ReactNode;
  onSelect: () => void;
  tone?: "danger";
}

/** A button that opens a short list of choices below it. Escape, a click elsewhere or a choice
 * closes it; arrow keys move between the choices. */
export function MenuButton({ label, items, kind = "ghost", disabled }: {
  label: string;
  items: MenuItem[];
  kind?: "primary" | "secondary" | "ghost" | "danger";
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        ref.current?.querySelector<HTMLElement>(".menu-trigger")?.focus();
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const all = Array.from(ref.current?.querySelectorAll<HTMLElement>("[role=menuitem]") ?? []);
        const at = all.indexOf(document.activeElement as HTMLElement);
        all[(at + (e.key === "ArrowDown" ? 1 : all.length - 1)) % all.length]?.focus();
      }
    };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", key);
    ref.current?.querySelector<HTMLElement>("[role=menuitem]")?.focus();
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  return (
    <div className="menu" ref={ref}>
      <button type="button" className={`btn btn-${kind} menu-trigger`} disabled={disabled} aria-haspopup="menu"
        aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>{label}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {open && (
        <div className="menu-list" role="menu" aria-label={label}>
          {items.map((it) => (
            <button key={it.label} type="button" role="menuitem" className={`menu-item ${it.tone ?? ""}`}
              onClick={() => { setOpen(false); it.onSelect(); }}>
              <span className="menu-label">{it.label}</span>
              {it.detail && <span className="menu-detail">{it.detail}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function Progress({ value, total, label }: { value?: number | null; total?: number | null; label?: string }) {
  const known = value != null && total != null && total > 0;
  const pctv = known ? Math.min(100, (value! / total!) * 100) : 0;
  return (
    <div className="progress" role="progressbar" aria-label={label} aria-valuenow={known ? Math.round(pctv) : undefined}>
      <div className={`progress-bar ${known ? "" : "indeterminate"}`} style={known ? { width: `${pctv}%` } : undefined} />
    </div>
  );
}

export function Switch({ checked, onChange, label, disabled }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <label className={`switch ${disabled ? "disabled" : ""}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch-track" aria-hidden="true">
        <span className="switch-thumb" />
      </span>
      <span>{label}</span>
    </label>
  );
}

export function Dialog({ title, onClose, children, footer }: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    ref.current?.querySelector<HTMLElement>("input, button, textarea")?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} ref={ref}>
        <header className="dialog-head">
          <h2>{title}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </header>
        <div className="dialog-body">{children}</div>
        {footer && <footer className="dialog-foot">{footer}</footer>}
      </div>
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}

export function Notice({ tone = "info", title, children, action }: {
  tone?: "info" | "warn" | "error" | "success";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className={`notice notice-${tone}`} role={tone === "error" ? "alert" : "status"}>
      <div className="notice-text">
        <strong>{title}</strong>
        {children && <div>{children}</div>}
      </div>
      {action && <div className="notice-action">{action}</div>}
    </div>
  );
}

export function Stat({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}
