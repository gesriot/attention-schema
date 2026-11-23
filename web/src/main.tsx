import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { LabApp } from "./components/lab/lab-app.tsx";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Не найден контейнер лаборатории.");
createRoot(root).render(
  <StrictMode>
    <LabApp />
  </StrictMode>,
);
