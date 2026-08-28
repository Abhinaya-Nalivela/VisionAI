import { useEffect, useState } from "react";

import {
  analyzeImage,
  getAnalysisHistory,
  deleteAnalysisHistoryItem,
  clearAnalysisHistory,
} from "./services/api";

import "./App.css";


function formatLabel(value) {
  if (!value) {
    return "";
  }

  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(
      /\b\w/g,
      (letter) => letter.toUpperCase()
    );
}


function getQualityClass(label) {
  if (label === "ACCEPTABLE") {
    return "quality-badge quality-acceptable";
  }

  if (label === "DEGRADED") {
    return "quality-badge quality-degraded";
  }

  if (label === "POTENTIALLY_DEFECTIVE") {
    return "quality-badge quality-defective";
  }

  return "quality-badge";
}


function formatStatistic(value) {
  if (typeof value !== "number") {
    return value;
  }

  if (Number.isInteger(value)) {
    return value;
  }

  return value.toFixed(4);
}


function App() {
  const [selectedFile, setSelectedFile] =
    useState(null);

  const [previewUrl, setPreviewUrl] =
    useState(null);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [history, setHistory] =
    useState([]);

  const [historyLoading, setHistoryLoading] =
    useState(true);

  const [historyActionLoading, setHistoryActionLoading] =
    useState(false);

  const [expandedHistoryId, setExpandedHistoryId] =
    useState(null);

  const [isDragging, setIsDragging] =
    useState(false);


  async function loadHistory() {
    try {
      setHistoryLoading(true);

      const data =
        await getAnalysisHistory();

      setHistory(data.items || []);
    } catch (err) {
      console.error(
        "Could not load history:",
        err
      );
    } finally {
      setHistoryLoading(false);
    }
  }


  useEffect(() => {
    loadHistory();
  }, []);


  function processFile(file) {
    if (!file) {
      return;
    }

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    const maxSize =
      10 * 1024 * 1024;

    if (!allowedTypes.includes(file.type)) {
      setError(
        "Unsupported image format. Use JPG, JPEG, PNG, or WEBP."
      );

      return;
    }

    if (file.size > maxSize) {
      setError(
        "Image exceeds the 10 MB size limit."
      );

      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(
        previewUrl
      );
    }

    setSelectedFile(file);

    setPreviewUrl(
      URL.createObjectURL(file)
    );

    setResult(null);
    setError("");
  }


  function handleFileChange(event) {
    const file =
      event.target.files[0];

    processFile(file);
  }


  function handleDragOver(event) {
    event.preventDefault();

    setIsDragging(true);
  }


  function handleDragLeave(event) {
    event.preventDefault();

    setIsDragging(false);
  }


  function handleDrop(event) {
    event.preventDefault();

    setIsDragging(false);

    const file =
      event.dataTransfer.files[0];

    processFile(file);
  }


  async function handleAnalyze() {
    if (!selectedFile) {
      setError(
        "Please select an image first."
      );

      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const data =
        await analyzeImage(
          selectedFile
        );

      setResult(data);

      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }


  async function handleDeleteHistory(
    analysisId
  ) {
    const confirmed =
      window.confirm(
        "Delete this analysis record?"
      );

    if (!confirmed) {
      return;
    }

    try {
      setHistoryActionLoading(true);

      await deleteAnalysisHistoryItem(
        analysisId
      );

      setHistory((currentHistory) =>
        currentHistory.filter(
          (item) =>
            item.analysis_id !==
            analysisId
        )
      );

      if (
        expandedHistoryId ===
        analysisId
      ) {
        setExpandedHistoryId(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setHistoryActionLoading(false);
    }
  }


  async function handleClearHistory() {
    if (history.length === 0) {
      return;
    }

    const confirmed =
      window.confirm(
        "Clear all analysis history? This action cannot be undone."
      );

    if (!confirmed) {
      return;
    }

    try {
      setHistoryActionLoading(true);

      await clearAnalysisHistory();

      setHistory([]);
      setExpandedHistoryId(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setHistoryActionLoading(false);
    }
  }


  function toggleHistoryDetails(
    analysisId
  ) {
    setExpandedHistoryId(
      (currentId) =>
        currentId === analysisId
          ? null
          : analysisId
    );
  }


  return (
    <div className="app">

      <header className="hero">
        <p className="eyebrow">
          AI-Powered Image Quality Assessment
        </p>

        <h1>
          VisionAI
        </h1>

        <p className="subtitle">
          Upload an image and evaluate its visual quality,
          degradation, and potential defects using computer
          vision and machine learning.
        </p>
      </header>


      <main className="container">

        <section className="card upload-card">

          <h2>
            Analyze Image
          </h2>

          <p className="section-description">
            JPG, JPEG, PNG, and WEBP files up to 10 MB.
          </p>


          <label
            className={
              isDragging
                ? "upload-box upload-box-dragging"
                : "upload-box"
            }

            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >

            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
            />

            <span className="upload-icon">
              ↑
            </span>

            <span className="upload-title">
              {isDragging
                ? "Drop image here"
                : "Choose or drag an image"}
            </span>

            <span className="upload-description">
              Click to browse or drag and drop your file here
            </span>

          </label>


          {previewUrl && (
            <div className="preview-wrapper">

              <img
                src={previewUrl}
                alt="Selected preview"
                className="preview-image"
              />

              <div className="file-info">

                <strong>
                  {selectedFile?.name}
                </strong>

                <span>
                  {(
                    selectedFile?.size /
                    (1024 * 1024)
                  ).toFixed(2)}
                  {" "}
                  MB
                </span>

              </div>

            </div>
          )}


          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={
              loading ||
              !selectedFile
            }
          >

            {loading
              ? "Analyzing..."
              : "Analyze Image"}

          </button>


          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

        </section>


        {result && (
          <section className="results-section">

            <div className="card score-card">

              <div>
                <p className="result-label">
                  Quality Score
                </p>

                <h2 className="score">
                  {result.quality_score}
                  <span>
                    /100
                  </span>
                </h2>
              </div>


              <div className="result-summary">

                <span
                  className={getQualityClass(
                    result.quality_label
                  )}
                >
                  {formatLabel(
                    result.quality_label
                  )}
                </span>

                <p>
                  Predicted class:{" "}
                  <strong>
                    {formatLabel(
                      result.predicted_class
                    )}
                  </strong>
                </p>

                <p>
                  Confidence:{" "}
                  <strong>
                    {(
                      result.confidence *
                      100
                    ).toFixed(1)}
                    %
                  </strong>
                </p>

                <p>
                  Severity:{" "}
                  <strong>
                    {formatLabel(
                      result.severity
                    )}
                  </strong>
                </p>

              </div>

            </div>


            <div className="card">

              <h2>
                Detected Issues
              </h2>

              {result.issues.length === 0 ? (

                <p className="success-message">
                  No major quality issues detected.
                </p>

              ) : (

                <div className="issues-list">

                  {result.issues.map(
                    (issue, index) => (

                      <div
                        className="issue-item"
                        key={`${issue.type}-${index}`}
                      >

                        <strong>
                          {formatLabel(
                            issue.type
                          )}
                        </strong>

                        <span>
                          {(
                            issue.confidence *
                            100
                          ).toFixed(1)}
                          %
                        </span>

                      </div>

                    )
                  )}

                </div>

              )}

            </div>


            <div className="card">

              <h2>
                Image Statistics
              </h2>

              <div className="stats-grid">

                {Object.entries(
                  result.image_statistics
                ).map(
                  ([name, value]) => (

                    <div
                      className="stat-item"
                      key={name}
                    >

                      <span>
                        {formatLabel(name)}
                      </span>

                      <strong>
                        {formatStatistic(
                          value
                        )}
                      </strong>

                    </div>

                  )
                )}

              </div>

            </div>

          </section>
        )}


        <section className="card history-card">

          <div className="history-header">

            <div>
              <h2>
                Analysis History
              </h2>

              <p className="section-description">
                Recently analyzed images
              </p>
            </div>


            <div className="history-header-actions">

              <span className="history-count">
                {history.length}
                {" "}
                records
              </span>

              <button
                type="button"
                className="clear-history-button"
                onClick={
                  handleClearHistory
                }
                disabled={
                  history.length === 0 ||
                  historyActionLoading
                }
              >
                Clear History
              </button>

            </div>

          </div>


          {historyLoading ? (

            <p className="history-empty">
              Loading history...
            </p>

          ) : history.length === 0 ? (

            <p className="history-empty">
              No analyses yet.
            </p>

          ) : (

            <div className="history-list">

              {history.map((item) => {

                const isExpanded =
                  expandedHistoryId ===
                  item.analysis_id;

                return (

                  <div
                    className="history-item-wrapper"
                    key={item.analysis_id}
                  >

                    <div className="history-item">

                      <div className="history-file">

                        <strong>
                          {item.filename}
                        </strong>

                        <span>
                          {new Date(
                            item.created_at
                          ).toLocaleString()}
                        </span>

                      </div>


                      <span
                        className={getQualityClass(
                          item.quality_label
                        )}
                      >
                        {formatLabel(
                          item.quality_label
                        )}
                      </span>


                      <div className="history-score">

                        <strong>
                          {item.quality_score}
                        </strong>

                        <span>
                          /100
                        </span>

                      </div>

                    </div>


                    <div className="history-actions">

                      <button
                        type="button"
                        className="history-details-button"
                        onClick={() =>
                          toggleHistoryDetails(
                            item.analysis_id
                          )
                        }
                      >
                        {isExpanded
                          ? "Read Less"
                          : "Read More"}
                      </button>


                      <button
                        type="button"
                        className="history-delete-button"
                        disabled={
                          historyActionLoading
                        }
                        onClick={() =>
                          handleDeleteHistory(
                            item.analysis_id
                          )
                        }
                      >
                        Delete
                      </button>

                    </div>


                    {isExpanded && (

                      <div className="history-details">

                        <div className="history-detail-summary">

                          <div>
                            <span>
                              Predicted Class
                            </span>

                            <strong>
                              {formatLabel(
                                item.predicted_class
                              )}
                            </strong>
                          </div>


                          <div>
                            <span>
                              Confidence
                            </span>

                            <strong>
                              {(
                                item.confidence *
                                100
                              ).toFixed(1)}
                              %
                            </strong>
                          </div>


                          <div>
                            <span>
                              Severity
                            </span>

                            <strong>
                              {formatLabel(
                                item.severity
                              )}
                            </strong>
                          </div>

                        </div>


                        <div className="history-detail-section">

                          <h3>
                            Detected Issues
                          </h3>

                          {item.issues?.length ? (

                            <div className="history-issue-list">

                              {item.issues.map(
                                (
                                  issue,
                                  index
                                ) => (

                                  <span
                                    key={`${item.analysis_id}-issue-${index}`}
                                  >
                                    {formatLabel(
                                      issue.type
                                    )}
                                    {" "}
                                    (
                                    {(
                                      issue.confidence *
                                      100
                                    ).toFixed(1)}
                                    %)
                                  </span>

                                )
                              )}

                            </div>

                          ) : (

                            <p>
                              No major quality issues detected.
                            </p>

                          )}

                        </div>


                        <div className="history-detail-section">

                          <h3>
                            Image Statistics
                          </h3>

                          <div className="history-stats-grid">

                            {Object.entries(
                              item.image_statistics ||
                              {}
                            ).map(
                              ([name, value]) => (

                                <div
                                  key={`${item.analysis_id}-${name}`}
                                  className="history-stat"
                                >

                                  <span>
                                    {formatLabel(
                                      name
                                    )}
                                  </span>

                                  <strong>
                                    {formatStatistic(
                                      value
                                    )}
                                  </strong>

                                </div>

                              )
                            )}

                          </div>

                        </div>

                      </div>

                    )}

                  </div>

                );

              })}

            </div>

          )}

        </section>

      </main>

    </div>
  );
}


export default App;