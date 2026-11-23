import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Eye,
  EyeOff,
  FlaskConical,
  Layers3,
  Pause,
  Play,
  RotateCcw,
  SkipForward,
  StepForward,
  TriangleAlert,
} from "lucide-react";
import {
  advanceLabEpisode,
  createLabEpisode,
  defaultSettings,
  labView,
  noiseLevels,
  type LabCondition,
  type LabEpisode,
  type LabSettings,
  type LabView,
} from "../../lib/lab/episode.ts";
import { fetchModel, type LoadedModel } from "../../lib/lab/model.ts";
import { PixelField, TruthField } from "./stage.tsx";
import { ActionValues } from "./action-values.tsx";

const conditions: { value: LabCondition; label: string; hint: string }[] = [
  {
    value: "schema",
    label: "Со схемой",
    hint: "Во втором канале – квадрат в положении окна внимания.",
  },
  {
    value: "none",
    label: "Без схемы",
    hint: "Второй канал пуст. Окно внимания существует и без схемы.",
  },
  {
    value: "commands",
    label: "История команд",
    hint: "Во втором канале – метка предыдущего действия, а не положение окна.",
  },
];
const interventions = [
  {
    value: "normal",
    label: "Без вмешательства",
    hint: "Наблюдение и внимание работают как при обучении.",
  },
  {
    value: "windowOff",
    label: "Окно выключено",
    hint: "Окно движется, но больше не убирает шум. Второй канал не меняется.",
  },
  {
    value: "fieldOff",
    label: "Поле скрыто",
    hint: "Обнулён второй канал во всей истории. Зрительное поле и окно сохранены.",
  },
  {
    value: "falseSchema",
    label: "Ложная схема",
    hint: "На одном шаге квадрат сдвигается во всей истории наблюдения. Реальное окно остаётся на месте.",
  },
] as const;
const score = (value: number) => value.toFixed(2).replace(".", ",");
const initialSeed = 4000000001;

export function LabApp() {
  const [tab, setTab] = useState<"level0" | "target">("level0");
  const [settings, setSettings] = useState<LabSettings>({ ...defaultSettings });
  const [loaded, setLoaded] = useState<LoadedModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [seed, setSeed] = useState(initialSeed);
  const [seedText, setSeedText] = useState(String(initialSeed));
  const [seedError, setSeedError] = useState("");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [showTruth, setShowTruth] = useState(false);
  const [view, setView] = useState<LabView | null>(null);
  const [totals, setTotals] = useState({ episodes: 0, tracking: 0, catching: 0 });
  const episodeRef = useRef<LabEpisode | null>(null);

  useEffect(() => {
    const abort = new AbortController();
    setLoading(true);
    setLoaded(null);
    setView(null);
    setRunning(false);
    setError(null);
    episodeRef.current = null;
    fetchModel(import.meta.env.BASE_URL, settings.condition, abort.signal)
      .then((model) => {
        if (!abort.signal.aborted) {
          setLoaded(model);
          setLoading(false);
        }
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted) {
          setLoading(false);
          setError(
            reason instanceof Error &&
              !(reason instanceof TypeError) &&
              !(reason instanceof SyntaxError)
              ? reason.message
              : "Не удалось прочитать файлы модели. Проверьте подключение и повторите загрузку.",
          );
        }
      });
    return () => abort.abort();
  }, [settings.condition, retry]);

  useEffect(() => {
    setRunning(false);
    setTotals({ episodes: 0, tracking: 0, catching: 0 });
    if (!loaded || loaded.metadata.condition !== settings.condition) return;
    try {
      const episode = createLabEpisode(loaded.model, settings, seed);
      episodeRef.current = episode;
      setView(labView(episode));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось начать эпизод.");
    }
  }, [loaded, settings]);

  const newEpisode = (nextSeed: number) => {
    if (!loaded || loaded.metadata.condition !== settings.condition) return;
    const episode = createLabEpisode(loaded.model, settings, nextSeed);
    episodeRef.current = episode;
    setSeed(nextSeed);
    setSeedText(String(nextSeed));
    setSeedError("");
    setView(labView(episode));
  };
  const advance = () => {
    const episode = episodeRef.current;
    if (!episode) return;
    if (episode.world.t === 10) {
      newEpisode((episode.seed + 1) % 4294967296);
      return;
    }
    advanceLabEpisode(episode);
    setView(labView(episode));
    if (episode.world.t === 10)
      setTotals((old) => ({
        episodes: old.episodes + 1,
        tracking: old.tracking + episode.world.trackingScore,
        catching: old.catching + episode.world.catchScore,
      }));
  };
  useEffect(() => {
    if (!running || tab !== "level0") return;
    const timer = window.setInterval(advance, 1000 / speed);
    return () => window.clearInterval(timer);
  }, [running, speed, tab, loaded, settings]);

  const changeSettings = (next: Partial<LabSettings>) => {
    setRunning(false);
    setSettings((previous) => ({ ...previous, ...next }));
  };
  const metadata = loaded?.metadata;
  const condition = conditions.find((item) => item.value === settings.condition)!;
  const intervention = interventions.find((item) => item.value === settings.intervention)!;
  const ready =
    !loading && !error && view !== null && loaded?.metadata.condition === settings.condition;
  return (
    <main className="lab-shell">
      <header className="site-header">
        <a className="brand" href={import.meta.env.BASE_URL} aria-label="Схема внимания – начало">
          <span className="brand-symbol">
            <Layers3 size={21} />
          </span>
          <span>Исследовательский стенд</span>
        </a>
        <a
          className="source-link"
          href="https://github.com/gesriot/attention-schema"
          target="_blank"
          rel="noreferrer"
        >
          Код и результаты <ArrowUpRight size={15} />
        </a>
      </header>
      <section className="intro">
        <div>
          <p className="eyebrow">ВОСПРИЯТИЕ · МОДЕЛЬ · УПРАВЛЕНИЕ</p>
          <h1>Схема внимания</h1>
          <p className="intro-copy">
            Внимание меняет то, что видит агент.
            <br className="desktop-break" /> А что меняет его представление о внимании?
          </p>
        </div>
        <div className="intro-note">
          <FlaskConical size={19} />
          <p>
            Меняйте условия.
            <br />
            Наблюдайте за действиями.
          </p>
        </div>
      </section>
      <div className="tabs" role="tablist" aria-label="Мир лаборатории">
        <button
          role="tab"
          id="level0-tab"
          aria-controls="lab-content"
          aria-selected={tab === "level0"}
          className={tab === "level0" ? "active" : ""}
          onClick={() => {
            setTab("level0");
            setRunning(false);
          }}
        >
          <span>01</span> Уровень 0
        </button>
        <button
          role="tab"
          id="target-tab"
          aria-controls="lab-content"
          aria-selected={tab === "target"}
          className={tab === "target" ? "active" : ""}
          onClick={() => {
            setTab("target");
            setRunning(false);
          }}
        >
          <span>02</span> Целевой мир <small>Скоро</small>
        </button>
      </div>
      <div
        id="lab-content"
        role="tabpanel"
        aria-labelledby={tab === "level0" ? "level0-tab" : "target-tab"}
      >
        {tab === "target" ? (
          <section className="coming-soon panel">
            <Activity size={32} />
            <p className="eyebrow">СЛЕДУЮЩИЙ ЭТАП</p>
            <h2>Целевой мир</h2>
            <p>Скоро</p>
            <button className="button" onClick={() => setTab("level0")}>
              Вернуться к уровню 0
            </button>
          </section>
        ) : (
          <>
            <div className="section-heading">
              <div>
                <h2>Поймать мяч сквозь шум</h2>
                <p>Постановка Уилтерсона и Грациано · агент DQN</p>
              </div>
              <div className="world-facts">
                <span>
                  Поле <b>10 × 10</b>
                </span>
                <span>
                  Эпизод <b>10 кадров</b>
                </span>
              </div>
            </div>
            {metadata && (
              <div
                className={`model-note ${metadata.iterations < 1500 ? "warning" : ""}`}
                role="status"
              >
                <TriangleAlert size={17} />
                <span>
                  {metadata.iterations < 1500 ? "Пробные веса" : "Загруженная модель"} · итераций:{" "}
                  {metadata.iterations.toLocaleString("ru-RU")} · субъект {metadata.subject}.{" "}
                  {metadata.iterations < 1500
                    ? "Короткое обучение для проверки лаборатории; это не воспроизведение опубликованных результатов."
                    : "Эпизоды на экране – отдельная демонстрация, а не итог исследовательского прогона."}
                </span>
              </div>
            )}
            <div className="lab-layout">
              <aside className="panel controls-panel" aria-label="Настройки эпизода">
                <div className="panel-heading">
                  <h2>Условия опыта</h2>
                  <span className="section-number">01</span>
                </div>
                <fieldset>
                  <legend>Вход агента</legend>
                  <div className="condition-options">
                    {conditions.map((item) => (
                      <button
                        key={item.value}
                        className={settings.condition === item.value ? "selected" : ""}
                        aria-pressed={settings.condition === item.value}
                        onClick={() =>
                          changeSettings({
                            condition: item.value,
                            ...(item.value !== "schema" && settings.intervention === "falseSchema"
                              ? { intervention: "normal" }
                              : {}),
                          })
                        }
                      >
                        <span className="radio-dot" />
                        {item.label}
                      </button>
                    ))}
                  </div>
                  <p className="control-hint">{condition.hint}</p>
                </fieldset>
                <fieldset>
                  <legend className="range-heading">
                    <label htmlFor="noise">Шум поля</label>
                    <output>{Math.round(settings.noise * 100)}%</output>
                  </legend>
                  <input
                    id="noise"
                    type="range"
                    min="0"
                    max="4"
                    step="1"
                    value={noiseLevels.indexOf(settings.noise as (typeof noiseLevels)[number])}
                    aria-valuetext={`${Math.round(settings.noise * 100)} процентов`}
                    onChange={(event) =>
                      changeSettings({ noise: noiseLevels[Number(event.target.value)]! })
                    }
                  />
                  <div className="range-labels">
                    <span>0%</span>
                    <span>25%</span>
                    <span>50%</span>
                    <span>75%</span>
                    <span>100%</span>
                  </div>
                  {settings.noise !== 0.5 ? (
                    <p className="inline-warning">
                      Для этого шума модели нет. Используется агент, обученный при шуме 50%.
                    </p>
                  ) : (
                    <p className="control-hint">Шум при обучении – 50%.</p>
                  )}
                </fieldset>
                <fieldset>
                  <legend>
                    <label htmlFor="intervention">Вмешательство</label>
                  </legend>
                  <select
                    id="intervention"
                    value={settings.intervention}
                    onChange={(event) =>
                      changeSettings({
                        intervention: event.target.value as LabSettings["intervention"],
                      })
                    }
                  >
                    {interventions.map((item) => (
                      <option
                        key={item.value}
                        value={item.value}
                        disabled={item.value === "falseSchema" && settings.condition !== "schema"}
                      >
                        {item.label}
                      </option>
                    ))}
                  </select>
                  <p className="control-hint">{intervention.hint}</p>
                  {settings.intervention === "falseSchema" && (
                    <div className="shift-controls">
                      <label htmlFor="shift">
                        Сдвиг по столбцам{" "}
                        <b>
                          {settings.falseShift > 0 ? "+" : ""}
                          {settings.falseShift}
                        </b>
                      </label>
                      <input
                        id="shift"
                        type="range"
                        min="-3"
                        max="3"
                        value={settings.falseShift}
                        onChange={(event) =>
                          changeSettings({ falseShift: Number(event.target.value) })
                        }
                      />
                      <label htmlFor="false-step">На шаге</label>
                      <select
                        id="false-step"
                        value={settings.falseStep}
                        onChange={(event) =>
                          changeSettings({ falseStep: Number(event.target.value) })
                        }
                      >
                        {Array.from({ length: 9 }, (_, i) => (
                          <option key={i} value={i + 1}>
                            {i + 1}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </fieldset>
                <form
                  className="seed-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    const value = Number(seedText);
                    if (
                      !/^\d+$/.test(seedText) ||
                      !Number.isSafeInteger(value) ||
                      value > 4294967295
                    ) {
                      setSeedError("Введите целое число от 0 до 4294967295.");
                      return;
                    }
                    setRunning(false);
                    newEpisode(value);
                  }}
                >
                  <label htmlFor="seed">Зерно эпизода</label>
                  <div>
                    <input
                      id="seed"
                      inputMode="numeric"
                      value={seedText}
                      onChange={(event) => setSeedText(event.target.value)}
                      aria-invalid={!!seedError}
                    />
                    <button
                      className="icon-button"
                      disabled={!ready}
                      title="Повторить с этим зерном"
                      aria-label="Повторить с этим зерном"
                    >
                      <RotateCcw size={17} />
                    </button>
                  </div>
                  {seedError && (
                    <p className="inline-warning" role="alert">
                      {seedError}
                    </p>
                  )}
                </form>
                <p className="micro-copy">
                  Изменение условий начинает новый эпизод и обнуляет средние.
                </p>
              </aside>
              <div className="observation-column">
                <section className="panel observation-panel" aria-label="Наблюдение агента">
                  <div className="panel-heading">
                    <h2>Что видит агент</h2>
                    <span className="frame-count">
                      Кадр <b>{view?.step ?? 1}</b> / 10
                    </span>
                  </div>
                  <div className="step-strip" aria-hidden="true">
                    {Array.from({ length: 10 }, (_, i) => (
                      <i key={i} className={(view?.step ?? 1) > i ? "seen" : ""} />
                    ))}
                  </div>
                  {loading ? (
                    <div className="loading-state" role="status">
                      <span className="loading-dot" />
                      Загружаем веса сети…
                    </div>
                  ) : error ? (
                    <div className="error-state" role="alert">
                      <TriangleAlert size={24} />
                      <p>{error}</p>
                      <button className="button" onClick={() => setRetry((n) => n + 1)}>
                        Повторить загрузку
                      </button>
                    </div>
                  ) : (
                    view && (
                      <>
                        <div className="agent-grids">
                          <div>
                            <h3>
                              <span className="legend-dot visual-dot" />
                              Зрительное поле
                            </h3>
                            <PixelField
                              pixels={view.frame.slice(0, 100)}
                              label="Зрительное поле агента: мяч и шум выглядят одинаково"
                            />
                            <p>
                              Мяч и шум выглядят одинаково.
                              <br />
                              {settings.intervention === "windowOff"
                                ? "Окно выключено: шум остаётся на всём поле."
                                : "Окно убирает шум на участке 3 × 3."}
                            </p>
                          </div>
                          <div>
                            <h3>
                              <span className="legend-dot schema-dot" />
                              {settings.condition === "commands" ? "Канал команд" : "Канал схемы"}
                            </h3>
                            <PixelField
                              pixels={view.frame.slice(100)}
                              schema
                              label={
                                settings.condition === "commands"
                                  ? "Канал предыдущей команды"
                                  : "Канал схемы агента"
                              }
                            />
                            <p>
                              {settings.intervention === "fieldOff"
                                ? "Второй канал обнулён во всей истории наблюдения."
                                : settings.condition === "commands"
                                  ? "Метка предыдущего действия. На первом кадре канал пуст."
                                  : settings.condition === "none"
                                    ? "Пустой канал: эта часть входа не сообщает агенту положение окна."
                                    : "Квадрат показывает положение окна. При подмене он может быть ложным."}
                            </p>
                          </div>
                        </div>
                        <div className="observation-footer">
                          <span>В памяти агента – последние 10 кадров.</span>
                          <button
                            className={`truth-toggle ${showTruth ? "enabled" : ""}`}
                            aria-pressed={showTruth}
                            onClick={() => setShowTruth((value) => !value)}
                          >
                            {showTruth ? <EyeOff size={16} /> : <Eye size={16} />}
                            {showTruth ? "Скрыть истину" : "Показать истину"}
                          </button>
                        </div>
                        {view.falseStatus === "outside" && (
                          <p className="inline-warning intervention-note" role="status">
                            Сдвиг не применён: квадрат в истории вышел бы за край. В проверке стенда
                            такие случаи исключаются.
                          </p>
                        )}
                        {view.falseStatus === "applied" && (
                          <p className="intervention-note" role="status">
                            {settings.falseShift === 0
                              ? "Выбран нулевой сдвиг: схема не изменена."
                              : `Подмена на шаге ${settings.falseStep} применена только к наблюдению этого шага.`}
                          </p>
                        )}
                      </>
                    )
                  )}
                </section>
                {showTruth && ready && view && (
                  <section className="panel truth-panel">
                    <div>
                      <p className="eyebrow">ДОСТУПНО ТОЛЬКО НАБЛЮДАТЕЛЮ</p>
                      <h2>Истина мира</h2>
                      <p>Эта панель не входит в наблюдение сети.</p>
                      <ul className="truth-legend">
                        <li>
                          <i className="ball-key" />
                          Мяч
                        </li>
                        <li>
                          <i className="window-key" />
                          Окно и его центр{!view.truth.windowActive && " (выключено)"}
                        </li>
                        <li>
                          <i className="paddle-key" />
                          Платформа
                        </li>
                        <li>
                          <i className="landing-key" />
                          Место приземления
                        </li>
                      </ul>
                    </div>
                    <TruthField truth={view.truth} />
                  </section>
                )}
                <section className="playback panel" aria-label="Управление воспроизведением">
                  <div className="playback-buttons">
                    <button
                      className="button primary"
                      disabled={!ready}
                      onClick={() => setRunning((value) => !value)}
                    >
                      {running ? <Pause size={17} /> : <Play size={17} />}
                      {running ? "Пауза" : "Пуск"}
                    </button>
                    <button
                      className="button"
                      disabled={!ready || view?.step === 10}
                      onClick={() => {
                        setRunning(false);
                        advance();
                      }}
                    >
                      <StepForward size={17} />
                      Шаг
                    </button>
                    <button
                      className="icon-button"
                      title="Новый эпизод"
                      aria-label="Новый эпизод"
                      disabled={!ready}
                      onClick={() => {
                        setRunning(false);
                        newEpisode((seed + 1) % 4294967296);
                      }}
                    >
                      <SkipForward size={18} />
                    </button>
                  </div>
                  <label className="speed-control" htmlFor="speed">
                    Скорость
                    <select
                      id="speed"
                      value={speed}
                      onChange={(event) => setSpeed(Number(event.target.value))}
                    >
                      {[1, 2, 4, 8].map((value) => (
                        <option key={value} value={value}>
                          {value} кадр/с
                        </option>
                      ))}
                    </select>
                  </label>
                </section>
                <section className="panel score-panel" aria-label="Счёт эпизода">
                  <div className="panel-heading">
                    <h2>Счёт эпизода</h2>
                    <span className="tag">{view?.step === 10 ? "Завершён" : "Текущий"}</span>
                  </div>
                  <div className="score-grid">
                    <div>
                      <span>Слежение</span>
                      <strong>
                        {score(view?.tracking ?? 0)}
                        <small> / 4</small>
                      </strong>
                      <p>±0,5 на кадрах 2–9</p>
                    </div>
                    <div>
                      <span>Ловля</span>
                      <strong>
                        {view?.step === 10 ? score(view.catching) : "···"}
                        <small> / 2</small>
                      </strong>
                      <p>±2 на кадре 10</p>
                    </div>
                    <div className="average-score">
                      <span>Среднее завершённых · {totals.episodes}</span>
                      <strong>
                        {totals.episodes ? score(totals.tracking / totals.episodes) : "···"}
                        <small> / </small>
                        {totals.episodes ? score(totals.catching / totals.episodes) : "···"}
                      </strong>
                      <p>Слежение / ловля при этих условиях</p>
                    </div>
                  </div>
                </section>
              </div>
              {view && ready ? (
                <ActionValues view={view} condition={settings.condition} />
              ) : (
                <section className="panel q-panel">
                  <div className="panel-heading">
                    <h2>Оценки действий</h2>
                    <span className="tag">Q</span>
                  </div>
                  <p className="small-copy">Оценки появятся после загрузки модели.</p>
                </section>
              )}
            </div>
            <section className="explanations">
              <article>
                <span>01 / МЕХАНИЗМ</span>
                <h3>Окно внимания</h3>
                <p>
                  Участок 3 × 3 очищает зрительное поле от шума. Агент перемещает его и платформу
                  одним из 16 действий. На первом кадре окно находится на мяче.
                </p>
              </article>
              <article>
                <span>02 / ПРЕДСТАВЛЕНИЕ</span>
                <h3>Схема внимания</h3>
                <p>
                  Отдельный канал показывает, где окно. В уровне 0 это заданная схема. Польза такого
                  канала сама по себе не доказывает, что агент построил модель внимания.
                </p>
              </article>
              <article>
                <span>03 / ИЗМЕРЕНИЕ</span>
                <h3>Что означает счёт</h3>
                <p>
                  Слежение отмечает мяч внутри окна, даже если очистка выключена. Ловля – мяч над
                  платформой в конце. Максимум за эпизод: 4 + 2 = 6.
                </p>
              </article>
            </section>
          </>
        )}
      </div>
      <footer className="site-footer">
        <span>Схема внимания · исследовательская лаборатория</span>
        <span>Наблюдаем поведение, не устанавливаем наличие переживаний.</span>
      </footer>
    </main>
  );
}
