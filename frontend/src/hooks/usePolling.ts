import { useEffect, useRef } from "react";

/**
 * Runs `cb` immediately and then every `intervalMs` while `enabled`.
 * Mirrors the original setInterval polling loops, with cleanup on unmount.
 */
export function usePolling(cb: () => void, intervalMs: number, enabled = true) {
  const saved = useRef(cb);
  saved.current = cb;

  useEffect(() => {
    if (!enabled) return;
    saved.current();
    const id = window.setInterval(() => saved.current(), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs, enabled]);
}
