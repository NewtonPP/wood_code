import type { Moisture } from "../types";

interface Props {
  classes: string[];
  meanProbs: Moisture["mean_probs"];
}

// Ported from renderMoistureProbs in the original index.html.
export default function MoistureBars({ classes, meanProbs }: Props) {
  if (!classes || classes.length === 0 || !meanProbs) {
    return <div className="prob-bars" id="moist-probs" />;
  }
  const get = (cls: string): number => {
    if (Array.isArray(meanProbs)) return 0; // labelled object expected; keep parity with original (indexes by class name)
    const v = (meanProbs as Record<string, number>)[cls];
    return v != null ? v : 0;
  };

  return (
    <div className="prob-bars" id="moist-probs">
      {classes.map((cls) => {
        const p = get(cls);
        return (
          <div className="prob-row" key={cls}>
            <div>{cls}</div>
            <div className="bar">
              <div style={{ width: Math.max(0, Math.min(100, p * 100)).toFixed(0) + "%" }} />
            </div>
            <div>{(p * 100).toFixed(0)}%</div>
          </div>
        );
      })}
    </div>
  );
}
