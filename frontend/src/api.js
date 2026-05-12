/** Production API origin; no trailing slash (paths like `/api/health` are appended). */
const rawApiBase = import.meta.env.VITE_API_BASE_URL ?? "";
export const API_BASE =
  typeof rawApiBase === "string" ? rawApiBase.trim().replace(/\/+$/, "") : "";

let getAccessToken = () => null;
let onUnauthorized = () => {};

export function configureApiClient({ getToken, onAuthError } = {}) {
  if (typeof getToken === "function") {
    getAccessToken = getToken;
  }
  if (typeof onAuthError === "function") {
    onUnauthorized = onAuthError;
  }
}

function withAuthHeaders(headers) {
  const h = new Headers();
  if (headers instanceof Headers) {
    headers.forEach((value, key) => h.set(key, value));
  } else if (headers && typeof headers === "object") {
    for (const [key, value] of Object.entries(headers)) {
      if (value != null) h.set(key, String(value));
    }
  }
  const token = getAccessToken();
  if (token) {
    h.set("Authorization", `Bearer ${token}`);
  }
  return h;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: withAuthHeaders(options.headers),
  });
  if (response.status === 401) {
    try {
      onUnauthorized();
    } catch {
      /* ignore */
    }
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

async function requestBlob(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: withAuthHeaders(options.headers),
  });
  if (response.status === 401) {
    try {
      onUnauthorized();
    } catch {
      /* ignore */
    }
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || "Request failed");
  }
  return {
    blob: await response.blob(),
    filename: getFilenameFromDisposition(response.headers.get("content-disposition")),
  };
}

function getFilenameFromDisposition(disposition) {
  const value = String(disposition || "");
  const utf8Match = value.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    return decodeURIComponent(utf8Match[1]);
  }
  const quotedMatch = value.match(/filename="([^"]+)"/i);
  if (quotedMatch?.[1]) {
    return quotedMatch[1];
  }
  const plainMatch = value.match(/filename=([^;]+)/i);
  return plainMatch?.[1]?.trim() || "";
}

function downloadBlob(blob, filename, fallback = "quiz-export") {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || fallback;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export const api = {
  health: () => request("/api/health"),
  listNotebooks: (query = "") => {
    const params = new URLSearchParams();
    const trimmed = String(query || "").trim();
    if (trimmed) {
      params.set("q", trimmed);
    }
    const suffix = params.toString();
    return request(`/api/notebooks${suffix ? `?${suffix}` : ""}`);
  },
  createNotebook: (payload) =>
    request("/api/notebooks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getNotebook: (notebookId) => request(`/api/notebooks/${notebookId}`),
  patchNotebook: (notebookId, payload) =>
    request(`/api/notebooks/${notebookId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  patchRun: (notebookId, runId, payload) =>
    request(`/api/notebooks/${notebookId}/runs/${runId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  deleteRun: (notebookId, runId) =>
    request(`/api/notebooks/${notebookId}/runs/${runId}`, {
      method: "DELETE",
    }),
  exportRun: async (notebookId, runId, { format = "zip", dataFormat = "json" } = {}) => {
    const params = new URLSearchParams();
    params.set("format", format);
    if (format === "zip") {
      params.set("data_format", dataFormat);
    }
    const query = params.toString();
    const path = `/api/notebooks/${notebookId}/runs/${runId}/export${query ? `?${query}` : ""}`;
    const { blob, filename } = await requestBlob(path);
    const fallback = format === "pdf" ? "quiz-export.pdf" : "quiz-export.zip";
    downloadBlob(blob, filename, fallback);
  },
  uploadSource: (notebookId, { file, title }) => {
    const formData = new FormData();
    formData.append("file", file);
    if (title) {
      formData.append("title", title);
    }
    return request(`/api/notebooks/${notebookId}/sources`, {
      method: "POST",
      body: formData,
    });
  },
  sendMessage: (notebookId, content) =>
    request(`/api/notebooks/${notebookId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    }),
  generateQuiz: (notebookId, payload) =>
    request(`/api/notebooks/${notebookId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  listQuizLists: () => request("/api/quiz-lists"),
  createQuizList: (payload) =>
    request("/api/quiz-lists", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getQuizList: (listId) => request(`/api/quiz-lists/${listId}`),
  addQuizListItem: (listId, payload) =>
    request(`/api/quiz-lists/${listId}/items`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  removeQuizListItem: (listId, itemId) =>
    request(`/api/quiz-lists/${listId}/items/${itemId}`, {
      method: "DELETE",
    }),
};
