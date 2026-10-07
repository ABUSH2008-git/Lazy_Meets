"""Ready-made term lists. Picking a domain adds its terms to the glossary, which helps
Whisper spell them right and tells the refinement model what words to expect."""
from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "Software engineering": [
        "API", "REST", "GraphQL", "gRPC", "microservice", "Kubernetes", "K8s", "Docker", "Helm", "Terraform",
        "CI/CD", "GitHub", "GitLab", "pull request", "PR", "merge conflict", "staging", "production", "rollback",
        "feature flag", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "RabbitMQ", "Elasticsearch",
        "AWS", "S3", "EC2", "Lambda", "GCP", "Azure", "CDN", "Nginx", "load balancer", "latency", "p50", "p95", "p99",
        "SLA", "SLO", "SLI", "on-call", "runbook", "postmortem", "Grafana", "Prometheus", "Datadog", "Sentry",
        "PagerDuty", "Jira", "sprint", "backlog", "standup", "OAuth", "JWT", "SSO", "TypeScript", "JavaScript",
        "React", "Next.js", "Node.js", "Python", "Django", "FastAPI", "Go", "Rust", "regression", "QA", "unit test",
        "integration test", "TTL", "cache", "webhook", "SDK", "CLI", "YAML", "JSON",
    ],
    "Machine learning / AI": [
        "PyTorch", "TensorFlow", "JAX", "Hugging Face", "transformer", "LLM", "GPT", "BERT", "Llama", "Whisper",
        "fine-tuning", "LoRA", "QLoRA", "RAG", "embeddings", "vector database", "inference", "GPU", "CUDA",
        "A100", "H100", "TPU", "epoch", "batch size", "learning rate", "overfitting", "dropout", "F1 score",
        "precision", "recall", "ROC AUC", "BLEU", "WER", "tokenizer", "checkpoint", "dataset", "ONNX",
        "quantization", "YOLO", "OpenCV", "ResNet", "ViT", "diffusion", "Stable Diffusion", "MLOps", "Kaggle",
        "XGBoost", "scikit-learn", "pandas", "NumPy", "Jupyter", "Colab", "hyperparameter", "benchmark",
    ],
    "Product & business": [
        "OKR", "KPI", "roadmap", "MVP", "go-to-market", "GTM", "ARR", "MRR", "churn", "CAC", "LTV", "NPS",
        "A/B test", "funnel", "conversion rate", "stakeholder", "Q1", "Q2", "Q3", "Q4", "retro", "PRD", "Figma",
        "user story", "onboarding", "SaaS", "B2B", "B2C", "ROI", "SKU", "SEO", "CRM", "Salesforce", "HubSpot",
        "Notion", "Slack", "launch", "beta", "pricing tier",
    ],
    "Finance": [
        "EBITDA", "P&L", "CapEx", "OpEx", "burn rate", "runway", "cash flow", "forecast", "invoice",
        "accounts payable", "accounts receivable", "GAAP", "IFRS", "audit", "YoY", "QoQ", "basis points",
        "valuation", "cap table", "term sheet", "Series A", "GST", "TDS", "fiscal year", "budget", "variance",
        "accrual", "depreciation", "working capital", "KYC", "AML", "UPI",
    ],
    "Healthcare": [
        "EHR", "EMR", "HIPAA", "ICU", "OPD", "MRI", "CT scan", "ECG", "triage", "comorbidity", "discharge summary",
        "prescription", "dosage", "milligrams", "clinical trial", "FDA", "telemedicine", "ICD-10", "radiology",
        "pathology", "hypertension", "diabetes",
    ],
    "College / student team": [
        "Inter IIT", "hackathon", "PS", "problem statement", "mid-eval", "end-eval", "deliverable", "GitHub repo",
        "README", "demo video", "deadline", "submission", "prof", "TA", "lab", "viva", "CPI", "SPI", "Moodle",
    ],
}


def terms_for(domains: list[str] | None) -> list[str]:
    out: list[str] = []
    for d in domains or []:
        out.extend(PRESETS.get(d, []))
    return out


def parse_terms(text: str) -> list[str]:
    """Split a free-text box into terms. Accepts commas, semicolons or new lines."""
    if not text:
        return []
    raw = text.replace(";", ",").replace("\n", ",").split(",")
    seen, out = set(), []
    for t in raw:
        t = t.strip().strip("•-*").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out
