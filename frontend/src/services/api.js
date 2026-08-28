const API_BASE_URL = "http://127.0.0.1:8001";


export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/api/analyze`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const errorData =
      await response.json().catch(() => null);

    throw new Error(
      errorData?.detail ||
      "Failed to analyze image"
    );
  }

  return response.json();
}


export async function getAnalysisHistory(
  limit = 20
) {
  const response = await fetch(
    `${API_BASE_URL}/api/history?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(
      "Failed to load analysis history"
    );
  }

  return response.json();
}


export async function deleteAnalysisHistoryItem(
  analysisId
) {
  const response = await fetch(
    `${API_BASE_URL}/api/history/${analysisId}`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    const errorData =
      await response.json().catch(() => null);

    throw new Error(
      errorData?.detail ||
      "Failed to delete history item"
    );
  }

  return response.json();
}


export async function clearAnalysisHistory() {
  const response = await fetch(
    `${API_BASE_URL}/api/history`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    const errorData =
      await response.json().catch(() => null);

    throw new Error(
      errorData?.detail ||
      "Failed to clear analysis history"
    );
  }

  return response.json();
}