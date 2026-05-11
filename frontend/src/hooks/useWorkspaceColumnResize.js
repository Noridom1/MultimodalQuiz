import { useCallback, useEffect, useRef, useState } from "react";

const STORAGE_KEY = "multimodalQuiz.workspace.columnWidths";

const DEFAULTS = { left: 300, right: 400 };
const LIMITS = { left: [220, 520], right: [260, 640], centerMin: 280, resizer: 8 };

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function readStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    return {
      left: clamp(Number(parsed.left) || DEFAULTS.left, LIMITS.left[0], LIMITS.left[1]),
      right: clamp(Number(parsed.right) || DEFAULTS.right, LIMITS.right[0], LIMITS.right[1]),
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function useWorkspaceColumnResize() {
  const [widths, setWidths] = useState(readStored);
  const widthsRef = useRef(widths);
  widthsRef.current = widths;

  const dragRef = useRef(null);
  const moveHandlerRef = useRef(null);
  const upHandlerRef = useRef(null);

  const persist = useCallback((next) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  }, []);

  const endDrag = useCallback(() => {
    const move = moveHandlerRef.current;
    const up = upHandlerRef.current;
    if (move) window.removeEventListener("mousemove", move);
    if (up) window.removeEventListener("mouseup", up);
    moveHandlerRef.current = null;
    upHandlerRef.current = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    const d = dragRef.current;
    dragRef.current = null;
    if (d?.lastWidths) persist(d.lastWidths);
  }, [persist]);

  const startDrag = useCallback(
    (which, gridEl) => (event) => {
      event.preventDefault();
      if (!gridEl) return;

      const startWidths = { ...widthsRef.current };
      dragRef.current = {
        which,
        startX: event.clientX,
        startLeft: startWidths.left,
        startRight: startWidths.right,
        gridEl,
        lastWidths: startWidths,
      };

      const onMove = (moveEvent) => {
        const d = dragRef.current;
        if (!d?.gridEl) return;
        const rect = d.gridEl.getBoundingClientRect();
        const total = rect.width;
        const reserved = LIMITS.resizer * 2 + LIMITS.centerMin;

        if (d.which === 0) {
          const dx = moveEvent.clientX - d.startX;
          const maxLeft = total - d.startRight - reserved;
          setWidths((w) => {
            const nextLeft = clamp(d.startLeft + dx, LIMITS.left[0], Math.max(LIMITS.left[0], maxLeft));
            const next = { left: nextLeft, right: w.right };
            dragRef.current.lastWidths = next;
            return next;
          });
        } else {
          const dx = moveEvent.clientX - d.startX;
          const maxRight = total - d.startLeft - reserved;
          setWidths((w) => {
            const nextRight = clamp(d.startRight - dx, LIMITS.right[0], Math.max(LIMITS.right[0], maxRight));
            const next = { left: w.left, right: nextRight };
            dragRef.current.lastWidths = next;
            return next;
          });
        }
      };

      moveHandlerRef.current = onMove;
      upHandlerRef.current = endDrag;
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", endDrag);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [endDrag],
  );

  useEffect(() => () => endDrag(), [endDrag]);

  return { widths, startDrag };
}
