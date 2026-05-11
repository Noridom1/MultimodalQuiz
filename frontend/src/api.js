const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

export const api = {
  health: () => request("/api/health"),
  listNotebooks: () => request("/api/notebooks"),
  createNotebook: (payload) =>
    request("/api/notebooks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getNotebook: (notebookId) => request(`/api/notebooks/${notebookId}`),
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
};

