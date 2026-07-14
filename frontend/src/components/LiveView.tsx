import { useEffect, useRef, useState } from "react";
import { frameUrl, ingestWsUrl } from "../lib/api";
import { usePolling } from "../hooks/usePolling";
import type { InferenceResult, IngestMessage } from "../types";

// Capture/stream tuning. We downsize to keep upload bandwidth low; monitoring
// rate (~6 fps) is plenty for wood-chip analysis (see deployment plan).
const CAPTURE_WIDTH = 640;
const CAPTURE_INTERVAL_MS = 160; // ~6 fps
const JPEG_QUALITY = 0.6;
const RECONNECT_MS = 1500;
const MAX_BUFFERED = 1_000_000; // skip a frame if the socket is backed up

type ConnState = "connecting" | "live" | "offline" | "no-camera" | "denied";

/**
 * Live feed. In the cloud role the browser captures its own USB camera, streams
 * downsized JPEG frames to the ingest WebSocket, and draws the AI overlay
 * locally from the returned JSON. In the device role it falls back to the
 * server-rendered annotated frame.
 */
export default function LiveView({ role }: { role: string }) {
  return role === "cloud" ? <BrowserCapture /> : <DeviceFrame />;
}

function DeviceFrame() {
  const [frame, setFrame] = useState<string>("");
  usePolling(() => setFrame(frameUrl()), 200);
  return <img id="video" src={frame} alt="Live camera feed" />;
}

const STATE_TEXT: Record<ConnState, string> = {
  connecting: "Connecting to camera…",
  live: "",
  offline: "Connection lost — retrying…",
  "no-camera": "No camera found",
  denied: "Camera permission denied",
};

function BrowserCapture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const resultRef = useRef<InferenceResult | null>(null);
  const [state, setState] = useState<ConnState>("connecting");
  const [aiError, setAiError] = useState<string | null>(null);
  const [uncalibrated, setUncalibrated] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;
    let captureTimer: number | undefined;
    let reconnectTimer: number | undefined;

    const getCapture = () => {
      if (!captureRef.current) captureRef.current = document.createElement("canvas");
      return captureRef.current;
    };

    const sendFrame = () => {
      const video = videoRef.current;
      const ws = wsRef.current;
      if (!video || !ws || ws.readyState !== WebSocket.OPEN) return;
      if (ws.bufferedAmount > MAX_BUFFERED) return; // backpressure: drop this frame
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) return;
      const cw = CAPTURE_WIDTH;
      const ch = Math.round((CAPTURE_WIDTH * vh) / vw);
      const canvas = getCapture();
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, cw, ch);
      canvas.toBlob(
        (blob) => {
          if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
          blob.arrayBuffer().then((buf) => {
            try {
              wsRef.current?.send(buf);
            } catch {
              /* socket closed between checks */
            }
          });
        },
        "image/jpeg",
        JPEG_QUALITY
      );
    };

    const drawOverlay = () => {
      const video = videoRef.current;
      const canvas = overlayRef.current;
      const result = resultRef.current;
      if (!video || !canvas) return;
      const dispW = video.clientWidth;
      const dispH = video.clientHeight;
      if (!dispW || !dispH) return;
      if (canvas.width !== dispW) canvas.width = dispW;
      if (canvas.height !== dispH) canvas.height = dispH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!result || !result.frame_size) return;

      // The video is letterboxed inside the panel (object-fit: contain), so
      // map model coordinates into the actual displayed video rect.
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) return;
      const scale = Math.min(dispW / vw, dispH / vh);
      const rw = vw * scale;
      const rh = vh * scale;
      const ox = (dispW - rw) / 2;
      const oy = (dispH - rh) / 2;
      const sx = rw / result.frame_size.w;
      const sy = rh / result.frame_size.h;

      // reference disk
      if (result.reference) {
        const { cx, cy, diameter_px } = result.reference;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(ox + cx * sx, oy + cy * sy, (diameter_px / 2) * sx, 0, Math.PI * 2);
        ctx.stroke();
      }

      for (const b of result.boxes || []) {
        const color = b.oversized ? "#ff3b3b" : "#22c55e";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(ox + b.x1 * sx, oy + b.y1 * sy, (b.x2 - b.x1) * sx, (b.y2 - b.y1) * sy);
        if (result.units === "mm") {
          ctx.fillStyle = color;
          ctx.font = "12px sans-serif";
          ctx.fillText(`${b.diameter.toFixed(0)}mm`, ox + b.x1 * sx + 2, Math.max(10, oy + b.y1 * sy - 3));
        }
      }
    };

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(ingestWsUrl());
      wsRef.current = ws;
      ws.onopen = () => {
        if (cancelled) return;
        setState("live");
        captureTimer = window.setInterval(sendFrame, CAPTURE_INTERVAL_MS);
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as IngestMessage;
          if (msg.ok && msg.result) {
            resultRef.current = msg.result;
            setAiError(null);
            setUncalibrated(msg.result.units !== "mm");
            window.requestAnimationFrame(drawOverlay);
          } else if (msg.error) {
            setAiError(msg.error);
          }
        } catch {
          /* ignore malformed message */
        }
      };
      const onDrop = () => {
        if (captureTimer) window.clearInterval(captureTimer);
        if (cancelled) return;
        setState("offline");
        reconnectTimer = window.setTimeout(connect, RECONNECT_MS);
      };
      ws.onclose = onDrop;
      ws.onerror = () => ws.close();
    };

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 } }, audio: false });
      } catch (e: unknown) {
        if (cancelled) return;
        const name = (e as DOMException)?.name;
        setState(name === "NotAllowedError" ? "denied" : "no-camera");
        return;
      }
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        try {
          await video.play();
        } catch {
          /* autoplay may require muted; the element is muted in JSX */
        }
      }
      connect();
    };

    start();

    return () => {
      cancelled = true;
      if (captureTimer) window.clearInterval(captureTimer);
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      const ws = wsRef.current;
      if (ws) {
        ws.onclose = null; // prevent reconnect on intentional teardown
        ws.close();
      }
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    // Fill .live-content so the #video absolute/inset-0 sizing has a real box
    // to resolve against (a static wrapper collapses to 0 height).
    <div className="live-video-wrap" style={{ position: "absolute", inset: 0 }}>
      <video
        id="video"
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: "100%", height: "100%", display: "block", objectFit: "contain" }}
      />
      <canvas
        ref={overlayRef}
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      />
      {state === "live" && !aiError && uncalibrated && (
        <div
          style={{
            position: "absolute",
            left: 8,
            top: 8,
            maxWidth: "70%",
            background: "rgba(180,120,20,0.85)",
            color: "#fff",
            fontSize: 12,
            padding: "4px 8px",
            borderRadius: 4,
            pointerEvents: "none",
          }}
        >
          Uncalibrated — sizes in pixels. Show the blue reference disk to enable mm sizing and
          oversize (red) highlighting.
        </div>
      )}
      {state === "live" && aiError && (
        <div
          style={{
            position: "absolute",
            left: 8,
            bottom: 8,
            maxWidth: "70%",
            background: "rgba(180,40,40,0.85)",
            color: "#fff",
            fontSize: 12,
            padding: "4px 8px",
            borderRadius: 4,
            pointerEvents: "none",
          }}
        >
          AI analysis unavailable: {aiError}
        </div>
      )}
      {state !== "live" && (
        <div
          className="live-overlay-status"
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.45)",
            color: "#fff",
            fontSize: 14,
            textAlign: "center",
            padding: 12,
          }}
        >
          {STATE_TEXT[state]}
        </div>
      )}
    </div>
  );
}
