// CellTrace Dive API client. Relative URLs ride the CRA proxy to :8000.
function req(path, options = {}) {
  return fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  }).then(async (res) => {
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data.detail ?? data.error ?? res.statusText;
      throw new Error(typeof d === "string" ? d : JSON.stringify(d));
    }
    return data;
  });
}

export function unwrap(payload) {
  return payload?.result !== undefined ? payload.result : payload;
}

function uploadFolder(kind, movieName, files, onProgress) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("movie_name", movieName);
    for (const f of files) {
      fd.append("files", f, f.name);
      fd.append("paths", f.webkitRelativePath || f.name);
    }
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload/${kind}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      try {
        const d = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(unwrap(d));
        else reject(new Error(d.detail || xhr.statusText));
      } catch (e) {
        reject(e);
      }
    };
    xhr.onerror = () => reject(new Error("upload failed (network)"));
    xhr.send(fd);
  });
}

export const api = {
  health: () => req("/api/health"),
  listDatasets: () => req("/api/module0/datasets"),
  loadDataset: (movie_id) =>
    req("/api/module0/load", { method: "POST", body: JSON.stringify({ movie_id }) }),
  validate: (movie_id) =>
    req("/api/module0/validate", { method: "POST", body: JSON.stringify({ movie_id }) }),
  extract: (movie_id, max_cells = 256) =>
    req("/api/module0/extract", { method: "POST", body: JSON.stringify({ movie_id, max_cells }) }),
  standardize: (movie_id, t, z) =>
    req(`/api/module0/standardize?movie_id=${encodeURIComponent(movie_id)}&t=${t}&z=${z}`, { method: "POST" }),
  tensorize: (movie_id, max_cells = 256) =>
    req("/api/module0/tensorize", { method: "POST", body: JSON.stringify({ movie_id, max_cells }) }),
  slice: (movie_id, t, z) =>
    req(`/api/module0/slice?movie_id=${encodeURIComponent(movie_id)}&t=${t}&z=${z}`),
  inspect: (movie_id, t, z) =>
    req(`/api/module0/inspect?movie_id=${encodeURIComponent(movie_id)}&t=${t}&z=${z}`),
  geffOverlay: (movie_id, t, z) =>
    req(`/api/module0/geff-overlay?movie_id=${encodeURIComponent(movie_id)}&t=${t}&z=${z}`),
  runModule0: (movie_id, max_cells = 256) =>
    req("/api/module0/run", { method: "POST", body: JSON.stringify({ movie_id, max_cells }) }),
  runCellDino: (movie_id, max_cells = 256) =>
    req("/api/module1/run", { method: "POST", body: JSON.stringify({ movie_id, max_cells }) }),
  embeddingMeta: (movie_id) => req(`/api/module1/embeddings/${encodeURIComponent(movie_id)}`),
  trainCellDino: (settings = {}) =>
    req("/api/module1/train", { method: "POST", body: JSON.stringify(settings) }),
  cellDinoTrainingStatus: () => req("/api/module1/training-status"),
  embeddingSimilarity: (movie_id) => req(`/api/module1/similarity/${encodeURIComponent(movie_id)}`),
  runModule2: (movie_id) =>
    req("/api/module2/run", { method: "POST", body: JSON.stringify({ movie_id }) }),
  module2GraphPreview: (movie_id, focusNodeId = null) => {
    const params = new URLSearchParams({ movie_id });
    if (focusNodeId !== null) params.set("focus_node_id", String(focusNodeId));
    return req(`/api/module2/graph-preview?${params.toString()}`);
  },
  runModule3: (movie_id, epochs = 50) =>
    req("/api/module3/run", { method: "POST", body: JSON.stringify({ movie_id, epochs }) }),
  runTracking: (movie_id, group_weights = null, tag = "") =>
    req("/api/module4/run", { method: "POST", body: JSON.stringify({ movie_id, group_weights, tag }) }),
  fitTrackingWeights: (train_movie_ids) =>
    req("/api/module4/fit-weights", { method: "POST", body: JSON.stringify({ train_movie_ids }) }),
  theatre: (movie_id, source = "fitted") =>
    req(`/api/module4/theatre?movie_id=${encodeURIComponent(movie_id)}&source=${source}`),
  geffCloud: (movie_id, max_points = 3000) =>
    req(`/api/journey/geff-cloud?movie_id=${encodeURIComponent(movie_id)}&max_points=${max_points}`),
  journeyArtifact: (name, movie_id = "") =>
    req(`/api/journey/artifact?name=${encodeURIComponent(name)}&movie_id=${encodeURIComponent(movie_id)}`),
  module3Attention: (movie_id, node_id, k = 8) =>
    req(`/api/journey/attention?movie_id=${encodeURIComponent(movie_id)}&node_id=${node_id}&k=${k}`),
  uploadZarr: (movieName, files, onProgress) => uploadFolder("zarr", movieName, files, onProgress),
  uploadGeff: (movieName, files, onProgress) => uploadFolder("geff", movieName, files, onProgress),
  uploadRegistry: () => req("/api/upload/registry"),
  deleteMovie: (movie_id) => req(`/api/upload/movie?movie_id=${encodeURIComponent(movie_id)}`, { method: "DELETE" }),
};

export default api;
