export function LoadingSkeleton({ lines = 4 }: { lines?: number }) {
  return (
    <div className="card">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skeleton skeleton-line" style={{ width: `${85 - i * 8}%` }} />
      ))}
    </div>
  );
}

export function EmptyState({ icon = "◇", title, detail }: { icon?: string; title: string; detail?: string }) {
  return (
    <div className="state-block">
      <div className="state-icon">{icon}</div>
      <div className="state-title">{title}</div>
      {detail && <div className="state-detail">{detail}</div>}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return <div className="error-banner">⚠ {message}</div>;
}
