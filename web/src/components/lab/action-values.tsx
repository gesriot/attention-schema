import { decodeAction } from "../../lib/level0/world.ts";
import { conditionConfig } from "../../lib/level0/dqn.ts";
import type { LabCondition, LabView } from "../../lib/lab/episode.ts";

const arrows = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"];
const directionNames = [
  "вниз",
  "вниз и вправо",
  "вправо",
  "вверх и вправо",
  "вверх",
  "вверх и влево",
  "влево",
  "вниз и влево",
];

export function actionLabel(condition: LabCondition, index: number): string {
  const action = decodeAction(conditionConfig(condition), index);
  return `Окно ${directionNames[action.attention]}, платформа ${action.paddle === 0 ? "влево" : "вправо"}`;
}

export function ActionValues({ view, condition }: { view: LabView; condition: LabCondition }) {
  const minimum = Math.min(0, ...view.values);
  const maximum = Math.max(0, ...view.values);
  const span = maximum - minimum || 1;
  const zero = (-minimum / span) * 100;
  return (
    <section className="panel q-panel" aria-label="Оценки действий">
      <div className="panel-heading">
        <h2>Оценки действий</h2>
        <span className="tag">Q</span>
      </div>
      <p className="small-copy">Чем выше оценка, тем предпочтительнее действие для сети.</p>
      {view.action === null ? (
        <div className="terminal-note">
          Эпизод завершён.
          <br />
          На кадре 10 действий нет.
        </div>
      ) : (
        <>
          <div className="q-head">
            <span>Окно / платформа</span>
            <span>Оценка</span>
          </div>
          <div className="q-list">
            {Array.from(view.values, (value, index) => {
              const action = decodeAction(conditionConfig(condition), index);
              const selected = view.action === index;
              const end = ((value - minimum) / span) * 100;
              return (
                <div
                  key={index}
                  className={`q-row ${selected ? "is-selected" : ""}`}
                  aria-label={`${actionLabel(condition, index)}: ${value.toFixed(4)}${selected ? ", выбрано" : ""}`}
                >
                  <span className="q-direction" aria-hidden="true">
                    {arrows[action.attention]}
                    <span>/</span>
                    {action.paddle === 0 ? "←" : "→"}
                  </span>
                  <span className="q-track">
                    <i className="q-zero" style={{ left: `${zero}%` }} />
                    <i
                      className="q-bar"
                      style={{
                        left: `${Math.min(zero, end)}%`,
                        width: `${Math.max(0.5, Math.abs(end - zero))}%`,
                      }}
                    />
                  </span>
                  <span className="q-number">{value.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
          <p className="chosen-action">
            <span className="status-dot" />
            {actionLabel(condition, view.action)}
          </p>
        </>
      )}
      <p className="micro-copy">
        8 сдвигов окна × 2 хода платформы. Стоять на месте нельзя; на границе движение
        ограничивается.
      </p>
    </section>
  );
}
