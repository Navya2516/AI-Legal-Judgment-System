import { useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000/api/judgment";

function App() {
  // ============================================================
  // CORE STATE
  // ============================================================

  const [selectedFile, setSelectedFile] = useState(null);
  const [documentId, setDocumentId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const [analysis, setAnalysis] = useState(null);
  const [documentInfo, setDocumentInfo] = useState(null);

  // ============================================================
  // ASK JUDGMENT
  // ============================================================

  const [question, setQuestion] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaResult, setQaResult] = useState(null);
  const [qaMessage, setQaMessage] = useState("");

  // ============================================================
  // SUMMARY
  // ============================================================

  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryMessage, setSummaryMessage] = useState("");

  // ============================================================
  // LEGAL PROVISION
  // ============================================================

  const [selectedProvision, setSelectedProvision] = useState("");
  const [provisionExplanation, setProvisionExplanation] = useState(null);
  const [provisionLoading, setProvisionLoading] = useState(false);
  const [provisionMessage, setProvisionMessage] = useState("");

  // ============================================================
  // MULTILINGUAL
  // ============================================================

  const [language, setLanguage] = useState("Tamil");
  const [multilingualExplanation, setMultilingualExplanation] =
    useState(null);
  const [multilingualLoading, setMultilingualLoading] = useState(false);
  const [multilingualMessage, setMultilingualMessage] = useState("");

  // ============================================================
  // HELPERS
  // ============================================================

  const toArray = (value) => {
    if (Array.isArray(value)) {
      return value;
    }

    if (
      value === null ||
      value === undefined ||
      value === ""
    ) {
      return [];
    }

    return [value];
  };

  const safeString = (value, fallback = "") => {
    if (
      value === null ||
      value === undefined
    ) {
      return fallback;
    }

    if (typeof value === "string") {
      return value;
    }

    if (
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      return String(value);
    }

    try {
      return JSON.stringify(value);
    } catch {
      return fallback;
    }
  };

  const getErrorMessage = async (response, fallback) => {
    try {
      const data = await response.json();

      if (typeof data?.detail === "string") {
        return data.detail;
      }

      if (
        data?.detail &&
        typeof data.detail === "object"
      ) {
        if (typeof data.detail.message === "string") {
          return data.detail.message;
        }

        return safeString(data.detail, fallback);
      }

      if (typeof data?.message === "string") {
        return data.message;
      }

      return fallback;
    } catch {
      return fallback;
    }
  };

  const formatItem = (item) => {
    if (
      item === null ||
      item === undefined
    ) {
      return "";
    }

    if (typeof item === "string") {
      return item;
    }

    if (
      typeof item === "number" ||
      typeof item === "boolean"
    ) {
      return String(item);
    }

    if (Array.isArray(item)) {
      return item
        .map((value) => formatItem(value))
        .join(", ");
    }

    if (typeof item === "object") {
      return Object.entries(item)
        .map(([key, value]) => {
          if (Array.isArray(value)) {
            return `${key}: ${value
              .map((v) => formatItem(v))
              .join(", ")}`;
          }

          if (
            typeof value === "object" &&
            value !== null
          ) {
            return `${key}: ${formatItem(value)}`;
          }

          return `${key}: ${String(value)}`;
        })
        .join(" | ");
    }

    return String(item);
  };

  const renderList = (items) => {
    const normalizedItems = toArray(items);

    if (normalizedItems.length === 0) {
      return null;
    }

    return (
      <ul className="ai-list">
        {normalizedItems.map((item, index) => (
          <li key={index}>
            {formatItem(item)}
          </li>
        ))}
      </ul>
    );
  };

  const renderEvidence = (evidence) => {
    if (
      evidence === null ||
      evidence === undefined ||
      evidence === ""
    ) {
      return null;
    }

    const evidenceItems = Array.isArray(evidence)
      ? evidence
      : [evidence];

    return (
      <div className="evidence-box">
        📖 <strong>Evidence:</strong>{" "}
        {evidenceItems.map((item, index) => (
          <span key={index}>
            {formatItem(item)}
            {index < evidenceItems.length - 1
              ? " • "
              : ""}
          </span>
        ))}
      </div>
    );
  };

  const clearToolResults = () => {
    setQuestion("");
    setQaResult(null);
    setQaMessage("");

    setSummary(null);
    setSummaryMessage("");

    setSelectedProvision("");
    setProvisionExplanation(null);
    setProvisionMessage("");

    setMultilingualExplanation(null);
    setMultilingualMessage("");
  };

  // ============================================================
  // FILE SELECTION
  // ============================================================

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      setSelectedFile(null);
      setMessage("Please select a PDF judgment first.");
      return;
    }

    const isPdf =
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf");

    if (!isPdf) {
      setSelectedFile(null);
      event.target.value = "";
      setMessage("Please select a valid PDF judgment file.");
      return;
    }

    setSelectedFile(file);

    setDocumentId(null);
    setAnalysis(null);
    setDocumentInfo(null);

    setMessage("");

    clearToolResults();
  };

  // ============================================================
  // ANALYZE JUDGMENT
  // ============================================================

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setMessage("Please select a PDF judgment first.");
      return;
    }

    setLoading(true);
    setMessage("Uploading judgment...");
    setAnalysis(null);
    setDocumentInfo(null);
    setDocumentId(null);

    clearToolResults();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // --------------------------------------------------------
      // STEP 1 — UPLOAD
      // --------------------------------------------------------

      const uploadResponse = await fetch(
        `${API_BASE}/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!uploadResponse.ok) {
        const errorMessage = await getErrorMessage(
          uploadResponse,
          "Judgment upload failed."
        );

        throw new Error(errorMessage);
      }

      const uploadData =
        await uploadResponse.json();

      const id = uploadData?.document_id;

      if (!id) {
        throw new Error(
          "Upload succeeded, but no document ID was returned."
        );
      }

      setDocumentId(id);

      setDocumentInfo({
        filename: uploadData.filename,
        total_pages: uploadData.total_pages,
        metadata: uploadData.metadata || {},
      });

      console.log(
        "Upload response:",
        uploadData
      );

      setMessage(
        "Judgment uploaded. Running legal analysis..."
      );

      // --------------------------------------------------------
      // STEP 2 — CONTINUATION / LEGAL ANALYSIS
      // --------------------------------------------------------

      const analysisResponse =
        await fetch(
          `${API_BASE}/${id}/continuation-insights`,
          {
            method: "POST",
          }
        );

      if (!analysisResponse.ok) {
        const errorMessage =
          await getErrorMessage(
            analysisResponse,
            "Judgment analysis failed."
          );

        throw new Error(errorMessage);
      }

      const analysisData =
        await analysisResponse.json();

      console.log(
        "Analysis response:",
        analysisData
      );

      setAnalysis(analysisData);

      setMessage(
        "Judgment analyzed successfully."
      );
    } catch (error) {
      console.error(
        "Analysis error:",
        error
      );

      setMessage(
        `Error: ${
          error?.message ||
          "Judgment analysis failed."
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // ASK THE JUDGMENT — RAG Q&A
  // ============================================================

  const handleAskQuestion = async () => {
    setQaMessage("");

    // Validation 1
    if (!documentId) {
      setQaMessage(
        "Analyze a judgment before asking a question."
      );
      setMessage(
        "Analyze a judgment before asking a question."
      );
      return;
    }

    // Validation 2
    if (!question.trim()) {
      setQaMessage(
        "Please enter a question."
      );
      setMessage(
        "Please enter a question."
      );
      return;
    }

    setQaLoading(true);
    setQaResult(null);

    try {
      const response = await fetch(
        `${API_BASE}/${documentId}/ask?question=${encodeURIComponent(
          question.trim()
        )}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        const errorMessage =
          await getErrorMessage(
            response,
            "Failed to answer the question."
          );

        throw new Error(errorMessage);
      }

      const data =
        await response.json();

      setQaResult(data);
      setQaMessage("");

      setMessage(
        "Question answered using evidence from the judgment."
      );
    } catch (error) {
      console.error(
        "Question error:",
        error
      );

      const errorMessage =
        error?.message ||
        "Failed to answer the question.";

      setQaMessage(
        `Error: ${errorMessage}`
      );

      setMessage(
        `Error: ${errorMessage}`
      );
    } finally {
      setQaLoading(false);
    }
  };

  // ============================================================
  // JUDGMENT SUMMARY
  // ============================================================

  const handleGenerateSummary = async () => {
    setSummaryMessage("");

    // Validation
    if (!documentId) {
      setSummaryMessage(
        "Analyze a judgment before generating the summary."
      );
      setMessage(
        "Analyze a judgment before generating the summary."
      );
      return;
    }

    setSummaryLoading(true);
    setSummary(null);

    try {
      const response = await fetch(
        `${API_BASE}/${documentId}/summary`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        const errorMessage =
          await getErrorMessage(
            response,
            "Failed to generate summary."
          );

        throw new Error(errorMessage);
      }

      const data =
        await response.json();

      if (!data?.summary) {
        throw new Error(
          "The server returned an empty summary."
        );
      }

      setSummary(data.summary);
      setSummaryMessage("");

      setMessage(
        "Evidence-grounded judgment summary generated."
      );
    } catch (error) {
      console.error(
        "Summary error:",
        error
      );

      const errorMessage =
        error?.message ||
        "Failed to generate summary.";

      setSummaryMessage(
        `Error: ${errorMessage}`
      );

      setMessage(
        `Error: ${errorMessage}`
      );
    } finally {
      setSummaryLoading(false);
    }
  };

  // ============================================================
  // LEGAL PROVISION EXPLANATION
  // ============================================================

  const handleProvisionExplanation = async () => {
    setProvisionMessage("");

    // Validation 1
    if (!documentId) {
      setProvisionMessage(
        "Analyze a judgment before explaining a provision."
      );
      setMessage(
        "Analyze a judgment before explaining a provision."
      );
      return;
    }

    // Validation 2
    if (!selectedProvision.trim()) {
      setProvisionMessage(
        "Please enter a legal provision, such as Article 368."
      );
      setMessage(
        "Please enter a legal provision, such as Article 368."
      );
      return;
    }

    setProvisionLoading(true);
    setProvisionExplanation(null);

    try {
      const response = await fetch(
        `${API_BASE}/${documentId}/provision-explanation?provision=${encodeURIComponent(
          selectedProvision.trim()
        )}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        const errorMessage =
          await getErrorMessage(
            response,
            "Provision was not found in the judgment."
          );

        throw new Error(errorMessage);
      }

      const data =
        await response.json();

      setProvisionExplanation(data);
      setProvisionMessage("");

      setMessage(
        "Evidence-grounded provision explanation generated."
      );
    } catch (error) {
      console.error(
        "Provision explanation error:",
        error
      );

      const errorMessage =
        error?.message ||
        "Failed to explain the provision.";

      setProvisionMessage(
        `Error: ${errorMessage}`
      );

      setMessage(
        `Error: ${errorMessage}`
      );
    } finally {
      setProvisionLoading(false);
    }
  };

  // ============================================================
  // MULTILINGUAL EXPLANATION
  // ============================================================

  const handleMultilingualExplanation = async () => {
    setMultilingualMessage("");

    // Validation 1
    if (!documentId) {
      setMultilingualMessage(
        "Analyze a judgment before generating a multilingual explanation."
      );
      setMessage(
        "Analyze a judgment before generating a multilingual explanation."
      );
      return;
    }

    // Validation 2
    if (!summary) {
      setMultilingualMessage(
        "Generate the Judgment Summary first."
      );
      setMessage(
        "Generate the Judgment Summary first. The multilingual endpoint uses the generated summary."
      );
      return;
    }

    setMultilingualLoading(true);
    setMultilingualExplanation(null);

    try {
      const response = await fetch(
        `${API_BASE}/${documentId}/multilingual-explanation?language=${encodeURIComponent(
          language
        )}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        const errorMessage =
          await getErrorMessage(
            response,
            "Failed to generate multilingual explanation."
          );

        throw new Error(errorMessage);
      }

      const data =
        await response.json();

      if (!data?.explanation) {
        throw new Error(
          "The server returned an empty multilingual explanation."
        );
      }

      setMultilingualExplanation(
        data.explanation
      );

      setMultilingualMessage("");

      setMessage(
        `${language} explanation generated successfully.`
      );
    } catch (error) {
      console.error(
        "Multilingual explanation error:",
        error
      );

      const errorMessage =
        error?.message ||
        "Failed to generate multilingual explanation.";

      setMultilingualMessage(
        `Error: ${errorMessage}`
      );

      setMessage(
        `Error: ${errorMessage}`
      );
    } finally {
      setMultilingualLoading(false);
    }
  };

  // ============================================================
  // ANALYSIS DATA
  // ============================================================

  const rankedProvisions = toArray(
    analysis?.ranked_provisions
  );

  const insights =
    analysis?.insights &&
    typeof analysis.insights === "object"
      ? analysis.insights
      : {};

  const potentialProvisions = toArray(
    insights?.potential_legal_provisions
  );

  const continuationIssues = toArray(
    insights?.issues_for_continuation
  );

  const relatedProvisions = toArray(
    insights?.related_provisions
  );

  // ============================================================
  // METADATA
  // ============================================================

  const metadata =
    documentInfo?.metadata || {};

  const totalPages =
    documentInfo?.total_pages || "—";

  const totalEvidence =
    rankedProvisions.reduce(
      (total, item) => {
        if (
          item &&
          typeof item === "object"
        ) {
          return (
            total +
            (Number(
              item.evidence_count
            ) || 0)
          );
        }

        return total;
      },
      0
    );

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">
        <div className="brand">

          <div className="brand-icon">
            ⚖️
          </div>

          <div>
            <h1>
              AI Legal Judgment
            </h1>

            <p>
              Evidence-Grounded Multilingual Legal Analysis
            </p>
          </div>

        </div>

        <div className="status">
          <span className="status-dot"></span>

          {loading
            ? "Analyzing..."
            : "System Ready"}
        </div>
      </header>

      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="container">

        {/* ====================================================
            HERO
        ==================================================== */}

        <section className="hero">

          <h2>
            Understand Court Judgments
            <br />

            <span>
              Clearly. Simply. With Evidence.
            </span>
          </h2>

          <p>
            Upload an Indian court judgment and let the
            system extract case information, identify
            relevant legal provisions, retrieve supporting
            evidence and generate multilingual explanations.
          </p>

        </section>

        {/* ====================================================
            UPLOAD
        ==================================================== */}

        <section className="upload-card">

          <div className="upload-header">

            <h3>
              📄 Upload Court Judgment
            </h3>

            <p>
              Start your legal judgment analysis by uploading a PDF.
            </p>

          </div>

          <div className="upload-box">

            <div className="upload-icon">
              📑
            </div>

            <h4>
              Select your judgment PDF
            </h4>

            <p>
              PDF files only • Indian court judgments
            </p>

            <input
              type="file"
              accept=".pdf,application/pdf"
              id="fileInput"
              onChange={handleFileChange}
            />

            <label
              htmlFor="fileInput"
              className="file-button"
            >
              Choose PDF
            </label>

            {selectedFile && (
              <p className="selected-file">
                📄 {selectedFile.name}
              </p>
            )}

          </div>

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading
              ? "⏳ Analyzing Judgment..."
              : "🔍 Analyze Judgment"}
          </button>

          {message && (
            <p className="message">
              {message}
            </p>
          )}

          {documentId && (
            <p className="document-id">
              Document ID:{" "}
              <strong>
                {documentId}
              </strong>
            </p>
          )}

        </section>

        {/* ====================================================
            ANALYSIS OVERVIEW
        ==================================================== */}

        <div className="section-title">

          <h3>
            📊 Analysis Overview
          </h3>

          <p>
            Key information generated from the legal analysis.
          </p>

        </div>

        <section className="stats">

          <div className="stat-card">
            <span className="label">
              Judgment Pages
            </span>

            <span className="number">
              {analysis
                ? totalPages
                : "—"}
            </span>
          </div>

          <div className="stat-card">
            <span className="label">
              Ranked Provisions
            </span>

            <span className="number">
              {analysis
                ? rankedProvisions.length
                : "—"}
            </span>
          </div>

          <div className="stat-card">
            <span className="label">
              Evidence Records
            </span>

            <span className="number">
              {analysis
                ? totalEvidence.toLocaleString()
                : "—"}
            </span>
          </div>

          <div className="stat-card">
            <span className="label">
              Continuation Issues
            </span>

            <span className="number">
              {analysis
                ? continuationIssues.length
                : "—"}
            </span>
          </div>

        </section>

        {/* ====================================================
            CASE INFORMATION
        ==================================================== */}

        <section className="card">

          <div className="section-title">

            <h3>
              ⚖️ Case Information
            </h3>

            <p>
              Case metadata extracted from the uploaded judgment.
            </p>

          </div>

          <div className="case-grid">

            <div className="case-item">
              <label>
                Case Number
              </label>

              <p>
                {safeString(
                  metadata.case_number,
                  "—"
                )}
              </p>
            </div>

            <div className="case-item">
              <label>
                Judgment Date
              </label>

              <p>
                {safeString(
                  metadata.judgment_date,
                  "—"
                )}
              </p>
            </div>

            <div className="case-item">
              <label>
                Petitioner
              </label>

              <p>
                {safeString(
                  metadata.petitioner,
                  "—"
                )}
              </p>
            </div>

            <div className="case-item">
              <label>
                Respondent
              </label>

              <p>
                {safeString(
                  metadata.respondent,
                  "—"
                )}
              </p>
            </div>

            <div className="case-item full">
              <label>
                Bench
              </label>

              <p>
                {safeString(
                  metadata.bench,
                  "—"
                )}
              </p>
            </div>

          </div>

        </section>

        {/* ====================================================
            AI TOOLS
            IMPORTANT:
            Kept visible even before document analysis so that
            validation messages can be tested.
        ==================================================== */}

        <section className="card ai-tools-card">

          <div className="section-title">

            <h3>
              🤖 Ask the Judgment
            </h3>

            <p>
              Ask a question and retrieve an evidence-grounded
              answer from the uploaded judgment.
            </p>

          </div>

          <div className="ai-control-row">

            <input
              className="ai-input"
              type="text"
              value={question}
              onChange={(event) => {
                setQuestion(
                  event.target.value
                );

                if (qaMessage) {
                  setQaMessage("");
                }
              }}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !qaLoading
                ) {
                  handleAskQuestion();
                }
              }}
              placeholder="Example: What is the basic structure doctrine?"
            />

            <button
              className="tool-button"
              onClick={handleAskQuestion}
              disabled={qaLoading}
            >
              {qaLoading
                ? "⏳ Answering..."
                : "Ask Question"}
            </button>

          </div>

          {qaMessage && (
            <p className="message validation-message">
              {qaMessage}
            </p>
          )}

          {qaResult && (
            <div className="ai-result">

              <h4>
                Answer
              </h4>

              <p className="ai-answer">
                {safeString(
                  qaResult.answer ||
                    qaResult.response ||
                    qaResult.result,
                  "No answer returned."
                )}
              </p>

              {qaResult.citations && (
                <div className="citation-row">
                  📄 Pages:{" "}
                  {Array.isArray(
                    qaResult.citations
                  )
                    ? qaResult.citations.join(", ")
                    : safeString(
                        qaResult.citations
                      )}
                </div>
              )}

              {qaResult.evidence &&
                renderEvidence(
                  qaResult.evidence
                )}

            </div>
          )}

        </section>

        {/* ====================================================
            JUDGMENT SUMMARY
        ==================================================== */}

        <section className="card ai-tools-card">

          <div className="section-title">

            <h3>
              📝 Judgment Summary
            </h3>

            <p>
              Generate a structured, evidence-grounded
              explanation of the uploaded judgment.
            </p>

          </div>

          <button
            className="tool-button"
            onClick={handleGenerateSummary}
            disabled={summaryLoading}
          >
            {summaryLoading
              ? "⏳ Generating Summary..."
              : "Generate Judgment Summary"}
          </button>

          {summaryMessage && (
            <p className="message validation-message">
              {summaryMessage}
            </p>
          )}

          {summary && (
            <div className="ai-result summary-result">

              {summary.case_overview && (
                <div className="ai-section">

                  <h4>
                    Case Overview
                  </h4>

                  <p>
                    {formatItem(
                      summary.case_overview
                    )}
                  </p>

                </div>
              )}

              <div className="ai-grid">

                {toArray(
                  summary.facts_of_case
                ).length > 0 && (
                  <div className="ai-section">

                    <h4>
                      Facts of the Case
                    </h4>

                    {renderList(
                      summary.facts_of_case
                    )}

                  </div>
                )}

                {toArray(
                  summary.legal_issues
                ).length > 0 && (
                  <div className="ai-section">

                    <h4>
                      Legal Issues
                    </h4>

                    {renderList(
                      summary.legal_issues
                    )}

                  </div>
                )}

              </div>

              {toArray(
                summary.important_arguments_or_content
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Important Arguments / Content
                  </h4>

                  {renderList(
                    summary.important_arguments_or_content
                  )}

                </div>
              )}

              {toArray(
                summary.court_reasoning
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Court Reasoning
                  </h4>

                  {renderList(
                    summary.court_reasoning
                  )}

                </div>
              )}

              {toArray(
                summary.decision_or_conclusion
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Decision / Conclusion
                  </h4>

                  {renderList(
                    summary.decision_or_conclusion
                  )}

                </div>
              )}

              {toArray(
                summary.important_legal_provisions
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Important Legal Provisions
                  </h4>

                  {renderList(
                    summary.important_legal_provisions
                  )}

                </div>
              )}

              {summary.simplified_explanation && (
                <div className="ai-section highlighted-section">

                  <h4>
                    Simple Explanation
                  </h4>

                  <p>
                    {formatItem(
                      summary.simplified_explanation
                    )}
                  </p>

                </div>
              )}

              {summary.limitations && (
                <div className="ai-section">

                  <h4>
                    Limitations
                  </h4>

                  <p>
                    {formatItem(
                      summary.limitations
                    )}
                  </p>

                </div>
              )}

            </div>
          )}

        </section>

        {/* ====================================================
            LEGAL PROVISION EXPLANATION
        ==================================================== */}

        <section className="card ai-tools-card">

          <div className="section-title">

            <h3>
              ⚖️ Explain a Legal Provision
            </h3>

            <p>
              Retrieve evidence-grounded information
              about a provision found in the judgment.
            </p>

          </div>

          <div className="ai-control-row">

            <input
              className="ai-input"
              type="text"
              list="provision-options"
              value={selectedProvision}
              onChange={(event) => {
                setSelectedProvision(
                  event.target.value
                );

                if (provisionMessage) {
                  setProvisionMessage("");
                }
              }}
              placeholder="Example: Article 368"
            />

            <datalist id="provision-options">

              {rankedProvisions.map(
                (item, index) => (
                  <option
                    key={index}
                    value={
                      typeof item === "object"
                        ? item.provision || ""
                        : String(item)
                    }
                  />
                )
              )}

            </datalist>

            <button
              className="tool-button"
              onClick={
                handleProvisionExplanation
              }
              disabled={provisionLoading}
            >
              {provisionLoading
                ? "⏳ Explaining..."
                : "Explain Provision"}
            </button>

          </div>

          {provisionMessage && (
            <p className="message validation-message">
              {provisionMessage}
            </p>
          )}

          {provisionExplanation?.explanation && (
            <div className="ai-result">

              <div className="provision-explanation-header">

                <h4>
                  {safeString(
                    provisionExplanation.provision ||
                      selectedProvision,
                    "Legal Provision"
                  )}
                </h4>

                {provisionExplanation
                  .explanation
                  .confidence && (
                  <span
                    className={`confidence ${String(
                      provisionExplanation
                        .explanation
                        .confidence
                    ).toLowerCase()}`}
                  >
                    {safeString(
                      provisionExplanation
                        .explanation
                        .confidence
                    )}
                  </span>
                )}

              </div>

              {provisionExplanation
                .explanation
                .what_the_provision_is_discussed_as && (
                <div className="ai-section">

                  <h4>
                    What it is discussed as
                  </h4>

                  <p>
                    {formatItem(
                      provisionExplanation
                        .explanation
                        .what_the_provision_is_discussed_as
                    )}
                  </p>

                </div>
              )}

              {provisionExplanation
                .explanation
                .why_it_appears && (
                <div className="ai-section">

                  <h4>
                    Why it appears
                  </h4>

                  <p>
                    {formatItem(
                      provisionExplanation
                        .explanation
                        .why_it_appears
                    )}
                  </p>

                </div>
              )}

              {provisionExplanation
                .explanation
                .court_discussion && (
                <div className="ai-section">

                  <h4>
                    Court Discussion
                  </h4>

                  <p>
                    {formatItem(
                      provisionExplanation
                        .explanation
                        .court_discussion
                    )}
                  </p>

                </div>
              )}

              {provisionExplanation
                .explanation
                .limitation && (
                <div className="ai-section">

                  <h4>
                    Limitation
                  </h4>

                  <p>
                    {formatItem(
                      provisionExplanation
                        .explanation
                        .limitation
                    )}
                  </p>

                </div>
              )}

              {renderEvidence(
                provisionExplanation
                  .explanation
                  .evidence
              )}

            </div>
          )}

        </section>

        {/* ====================================================
            MULTILINGUAL EXPLANATION
        ==================================================== */}

        <section className="card ai-tools-card">

          <div className="section-title">

            <h3>
              🌐 Multilingual Explanation
            </h3>

            <p>
              Generate the judgment explanation in a
              supported Indian language while preserving
              source page citations.
            </p>

          </div>

          <div className="ai-control-row">

            <select
              className="ai-input language-select"
              value={language}
              onChange={(event) =>
                setLanguage(
                  event.target.value
                )
              }
            >
              <option>English</option>
              <option>Tamil</option>
              <option>Hindi</option>
              <option>Telugu</option>
              <option>Malayalam</option>
              <option>Kannada</option>
            </select>

            <button
              className="tool-button"
              onClick={
                handleMultilingualExplanation
              }
              disabled={multilingualLoading}
            >
              {multilingualLoading
                ? "⏳ Generating..."
                : "Generate Explanation"}
            </button>

          </div>

          {!summary && (
            <p className="tool-note">
              Generate the Judgment Summary first.
              The multilingual endpoint uses the
              generated summary.
            </p>
          )}

          {multilingualMessage && (
            <p className="message validation-message">
              {multilingualMessage}
            </p>
          )}

          {multilingualExplanation && (
            <div className="ai-result multilingual-result">

              <h4>
                {safeString(
                  multilingualExplanation.language,
                  language
                )}{" "}
                Explanation
              </h4>

              {multilingualExplanation.case_overview && (
                <div className="ai-section">

                  <h4>
                    Case Overview
                  </h4>

                  <p>
                    {formatItem(
                      multilingualExplanation.case_overview
                    )}
                  </p>

                </div>
              )}

              {toArray(
                multilingualExplanation.facts_of_case
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Facts of the Case
                  </h4>

                  {renderList(
                    multilingualExplanation.facts_of_case
                  )}

                </div>
              )}

              {toArray(
                multilingualExplanation.legal_issues
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Legal Issues
                  </h4>

                  {renderList(
                    multilingualExplanation.legal_issues
                  )}

                </div>
              )}

              {toArray(
                multilingualExplanation
                  .important_arguments_or_content
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Important Arguments / Content
                  </h4>

                  {renderList(
                    multilingualExplanation
                      .important_arguments_or_content
                  )}

                </div>
              )}

              {toArray(
                multilingualExplanation.court_reasoning
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Court Reasoning
                  </h4>

                  {renderList(
                    multilingualExplanation
                      .court_reasoning
                  )}

                </div>
              )}

              {toArray(
                multilingualExplanation.decision_or_conclusion
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Decision / Conclusion
                  </h4>

                  {renderList(
                    multilingualExplanation
                      .decision_or_conclusion
                  )}

                </div>
              )}

              {toArray(
                multilingualExplanation
                  .important_legal_provisions
              ).length > 0 && (
                <div className="ai-section">

                  <h4>
                    Important Legal Provisions
                  </h4>

                  {renderList(
                    multilingualExplanation
                      .important_legal_provisions
                  )}

                </div>
              )}

              {multilingualExplanation
                .simplified_explanation && (
                <div className="ai-section highlighted-section">

                  <h4>
                    Simple Explanation
                  </h4>

                  <p>
                    {formatItem(
                      multilingualExplanation
                        .simplified_explanation
                    )}
                  </p>

                </div>
              )}

              {multilingualExplanation
                .limitations && (
                <div className="ai-section">

                  <h4>
                    Limitations
                  </h4>

                  <p>
                    {formatItem(
                      multilingualExplanation
                        .limitations
                    )}
                  </p>

                </div>
              )}

              {multilingualExplanation
                .disclaimer && (
                <div className="tool-note">
                  ⚠️{" "}
                  {formatItem(
                    multilingualExplanation
                      .disclaimer
                  )}
                </div>
              )}

            </div>
          )}

        </section>

        {/* ====================================================
            RELEVANT LEGAL PROVISIONS
        ==================================================== */}

        <section className="card">

          <div className="section-title">

            <h3>
              📚 Relevant Legal Provisions
            </h3>

            <p>
              Provisions ranked according to evidence
              and relevance within the judgment.
            </p>

          </div>

          {!analysis && (
            <div className="empty-state">
              Analyze a judgment to view relevant
              legal provisions.
            </div>
          )}

          {analysis &&
            rankedProvisions.length === 0 && (
              <div className="empty-state">
                No ranked legal provisions were returned.
              </div>
            )}

          {rankedProvisions.map(
            (item, index) => {

              const provision =
                item &&
                typeof item === "object"
                  ? item
                  : {
                      provision: item,
                    };

              return (
                <div
                  className="provision-card"
                  key={index}
                >

                  <div className="provision-top">

                    <div>

                      <span className="rank">
                        #{index + 1}
                      </span>

                      <h4>
                        {safeString(
                          provision.provision,
                          "Legal Provision"
                        )}
                      </h4>

                    </div>

                    <div className="score">
                      {provision.relevance_score !==
                      undefined
                        ? Number(
                            provision.relevance_score
                          ).toFixed(2)
                        : "—"}
                    </div>

                  </div>

                  <div className="provision-details">

                    <span
                      className={`confidence ${
                        provision.confidence
                          ? String(
                              provision.confidence
                            ).toLowerCase()
                          : ""
                      }`}
                    >
                      {safeString(
                        provision.confidence,
                        "Unknown"
                      )}
                    </span>

                    {provision.page_count !==
                      undefined && (
                      <span>
                        📄{" "}
                        {provision.page_count} pages
                      </span>
                    )}

                    {provision.evidence_count !==
                      undefined && (
                      <span>
                        🔎{" "}
                        {Number(
                          provision.evidence_count
                        ).toLocaleString()}{" "}
                        evidence records
                      </span>
                    )}

                  </div>

                </div>
              );
            }
          )}

        </section>

        {/* ====================================================
            POTENTIAL PROVISIONS
        ==================================================== */}

        <section className="card">

          <div className="section-title">

            <h3>
              💡 Potential Legal Provisions
            </h3>

            <p>
              Provisions identified as potentially
              relevant for continued legal research.
            </p>

          </div>

          {!analysis && (
            <div className="empty-state">
              Analyze the judgment to generate
              continuation-oriented legal insights.
            </div>
          )}

          {analysis &&
            potentialProvisions.length === 0 && (
              <div className="empty-state">
                No potential provisions were returned.
              </div>
            )}

          {potentialProvisions.map(
            (item, index) => {

              const provision =
                item &&
                typeof item === "object"
                  ? item
                  : {
                      provision: item,
                    };

              return (
                <div
                  className="insight-card"
                  key={index}
                >

                  <div className="insight-header">

                    <h4>
                      {safeString(
                        provision.provision,
                        "Legal Provision"
                      )}
                    </h4>

                    <span
                      className={`confidence ${
                        provision.confidence
                          ? String(
                              provision.confidence
                            ).toLowerCase()
                          : ""
                      }`}
                    >
                      {safeString(
                        provision.confidence,
                        "Unknown"
                      )}
                    </span>

                  </div>

                  <div className="insight-section">

                    <strong>
                      Why it may be relevant
                    </strong>

                    <p>
                      {formatItem(
                        provision.why_relevant ||
                          "Not provided."
                      )}
                    </p>

                  </div>

                  <div className="insight-section">

                    <strong>
                      Court discussion
                    </strong>

                    <p>
                      {formatItem(
                        provision.judgment_discussion ||
                          "Not provided."
                      )}
                    </p>

                  </div>

                  <div className="insight-section">

                    <strong>
                      Potential continuation issue
                    </strong>

                    <p>
                      {formatItem(
                        provision.potential_continuation_issue ||
                          "Not provided."
                      )}
                    </p>

                  </div>

                  {provision.evidence &&
                    renderEvidence(
                      provision.evidence
                    )}

                </div>
              );
            }
          )}

        </section>

        {/* ====================================================
            CONTINUATION ISSUES
        ==================================================== */}

        <section className="card">

          <div className="section-title">

            <h3>
              🔎 Issues for Continued Research
            </h3>

            <p>
              Issues identified from the supplied
              judgment evidence that may warrant
              further legal research.
            </p>

          </div>

          {!analysis && (
            <div className="empty-state">
              No continuation issues available yet.
            </div>
          )}

          {analysis &&
            continuationIssues.length === 0 && (
              <div className="empty-state">
                No continuation issues were returned.
              </div>
            )}

          {continuationIssues.map(
            (item, index) => {

              const issue =
                item &&
                typeof item === "object"
                  ? item
                  : {
                      issue: item,
                    };

              return (
                <div
                  className="issue-card"
                  key={index}
                >

                  <div className="issue-number">
                    {index + 1}
                  </div>

                  <div className="issue-content">

                    <h4>
                      {safeString(
                        issue.issue,
                        "Legal Issue"
                      )}
                    </h4>

                    <p>
                      {formatItem(
                        issue.description ||
                          "No description provided."
                      )}
                    </p>

                    {issue.evidence &&
                      renderEvidence(
                        issue.evidence
                      )}

                    <span
                      className={`confidence ${
                        issue.confidence
                          ? String(
                              issue.confidence
                            ).toLowerCase()
                          : ""
                      }`}
                    >
                      Confidence:{" "}
                      {safeString(
                        issue.confidence,
                        "Unknown"
                      )}
                    </span>

                  </div>

                </div>
              );
            }
          )}

        </section>

        {/* ====================================================
            RELATED PROVISIONS
        ==================================================== */}

        <section className="card">

          <div className="section-title">

            <h3>
              🔗 Related Legal Provisions
            </h3>

            <p>
              Connections between provisions identified
              in the judgment.
            </p>

          </div>

          {!analysis && (
            <div className="empty-state">
              Related provisions will appear
              after analysis.
            </div>
          )}

          {analysis &&
            relatedProvisions.length === 0 && (
              <div className="empty-state">
                No related provisions were returned.
              </div>
            )}

          {relatedProvisions.map(
            (item, index) => {

              const related =
                item &&
                typeof item === "object"
                  ? item
                  : {
                      provision: item,
                    };

              return (
                <div
                  className="related-card"
                  key={index}
                >

                  <h4>
                    {safeString(
                      related.provision,
                      "Legal Provision"
                    )}
                  </h4>

                  <p>
                    {formatItem(
                      related.relationship ||
                        "Relationship not provided."
                    )}
                  </p>

                  {related.evidence &&
                    renderEvidence(
                      related.evidence
                    )}

                </div>
              );
            }
          )}

        </section>

        {/* ====================================================
            DISCLAIMER
        ==================================================== */}

        {analysis && (
          <section className="card">

            <div className="disclaimer">

              ⚠️ <strong>Important:</strong>

              <br />

              {formatItem(
                insights.overall_limitation ||
                  "These insights are generated from the supplied judgment evidence and should be independently verified."
              )}

              <br />
              <br />

              {formatItem(
                insights.disclaimer ||
                  "These AI-generated insights are research support, not legal advice. A qualified legal professional should verify the information before relying on it."
              )}

            </div>

          </section>
        )}

      </main>

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer>
        AI-Powered Multilingual Legal Judgment
        Explanation System
      </footer>

    </div>
  );
}

export default App;