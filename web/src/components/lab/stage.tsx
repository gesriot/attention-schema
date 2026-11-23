import type { LabView } from "../../lib/lab/episode.ts";

export function PixelField({
  pixels,
  schema = false,
  label,
}: {
  pixels: Uint8Array;
  schema?: boolean;
  label: string;
}) {
  return (
    <svg className="pixel-field" viewBox="0 0 320 320" role="img" aria-label={label}>
      {Array.from({ length: 10 }, (_, i) => (
        <g key={i} className="axis-label">
          <text x={34 + i * 28} y={12} textAnchor="middle">
            {i + 1}
          </text>
          <text x={9} y={38 + i * 28} textAnchor="middle">
            {i + 1}
          </text>
        </g>
      ))}
      {Array.from(pixels, (value, i) => (
        <rect
          key={i}
          x={20 + (i % 10) * 28}
          y={20 + Math.floor(i / 10) * 28}
          width={27}
          height={27}
          fill={schema ? (value ? "#b5a6ff" : "#202a3a") : value ? "#0c1018" : "#8793a6"}
        />
      ))}
      <rect x="19.5" y="19.5" width="280" height="280" fill="none" stroke="#435069" />
    </svg>
  );
}

export function TruthField({ truth }: { truth: LabView["truth"] }) {
  const center = (n: number) => 20 + n * 28 + 13.5;
  return (
    <svg
      className="truth-field"
      viewBox="0 0 320 325"
      role="img"
      aria-label={`Истина: мяч в строке ${truth.ball.row + 1}, столбце ${truth.ball.col + 1}; приземление в столбце ${truth.landingCol + 1}`}
    >
      {Array.from({ length: 100 }, (_, i) => (
        <rect
          key={i}
          x={20 + (i % 10) * 28}
          y={20 + Math.floor(i / 10) * 28}
          width={27}
          height={27}
          fill="#202a3a"
        />
      ))}
      <rect
        x={20 + (truth.window.col - 1) * 28}
        y={20 + (truth.window.row - 1) * 28}
        width="83"
        height="83"
        fill={truth.windowActive ? "#89dfcf22" : "none"}
        stroke="#89dfcf"
        strokeWidth="2"
        strokeDasharray={truth.windowActive ? undefined : "5 4"}
      />
      <path
        d={`M${center(truth.window.col) - 5},${center(truth.window.row)}h10 M${center(truth.window.col)},${center(truth.window.row) - 5}v10`}
        stroke="#89dfcf"
        strokeWidth="1.5"
      />
      <rect x={20 + (truth.paddle - 1) * 28} y="290" width="83" height="8" rx="3" fill="#b5a6ff" />
      <circle cx={center(truth.ball.col)} cy={center(truth.ball.row)} r="8" fill="#ffc391" />
      <path d={`M${center(truth.landingCol)},304l-5,8h10z`} fill="#ffc391" />
    </svg>
  );
}
