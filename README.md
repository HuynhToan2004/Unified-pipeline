# Unified Pipeline (Phase 1 + Phase 2)

Pipeline hợp nhất để so sánh LLM reviewer vs Human reviewer theo nhiều chiều phân tích.

## Input dataset
- `dataset/Human_and_meta_reviews/*.json`
- `dataset/SEA_reviews/*.txt`

## Outputs
- `phase1/Phase1.jsonl`
- `phase2/Master_Classification.jsonl`
- `phase2/Novelty_Verification_Targets.jsonl`
- `phase3/output/Novelty_Assessment_Report.json`
- `phase4/output/Phase4_Metrics_Report.json`

## Cài đặt
```powershell
cd C:\Users\HOY9HC\Desktop\Code\Unified_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy end-to-end
```powershell
python scripts\run_unified_pipeline.py --paper-id 0A5o6dCKeK -v
```

## Chạy riêng từng phase
```powershell
python scripts\run_phase1.py --paper-id 0A5o6dCKeK -v
python scripts\run_phase2.py --phase1-input phase1\Phase1.jsonl -v
```

## Chạy Phase 2 mới trong `phase2/`
```powershell
cd C:\Users\HOY9HC\Desktop\Code\Unified_pipeline\phase2
python run_phase2.py --phase1-input ..\phase1\Phase1.jsonl --master-output ..\phase2\output\Master_Classification.jsonl --novelty-output ..\phase2\output\Novelty_Verification_Targets.jsonl --llm-provider heuristic -v
```

- Phase2 hiện dùng thiết kế **1 API call cho mỗi reviewer record** (gộp toàn bộ sections classification + novelty targets trong một lần gọi) để tiết kiệm token và giảm latency.

### Optional paper_abstract_intro input
- Có thể truyền `--paper-context-dir <dir>`.
- Mỗi paper đặt tại `<dir>/<paper_id>.txt` hoặc `<dir>/<paper_id>.json`.
- Nếu là JSON, ưu tiên các key: `paper_abstract_intro`, `abstract_intro`, `abstract`, `paper_text`.

## Phase 3 (Novelty-only)
Phase này chỉ lấy related work từ Semantic API để chuẩn bị cho Phase4 tính metrics novelty.
Không chạy CFI/aspect analysis và không chạy novelty judgement ở Phase3.
Mỗi item output được chuẩn hóa theo style Task2 (`mode`, `queries`, `candidate_pool_top30`, `stats`) để dễ reuse ở Phase4.

### Chạy cơ bản (heuristic)
```powershell
cd C:\Users\HOY9HC\Desktop\Code\Unified_pipeline\phase3
python run_phase3.py --phase2-novelty-input ..\phase2\output\Novelty_Verification_Targets.jsonl -v
```

Nếu môi trường proxy chặn Semantic Scholar, có thể tắt semantic API:

```powershell
$env:DISABLE_SEMANTIC_API="1"
python run_phase3.py
```

Output mặc định:
- `phase3/output/Novelty_Assessment_Report.json`
- `phase3/output/Novelty_Assessment_Report.md`

Schema chính trong `Novelty_Assessment_Report.json`:
- Top-level: `meta`, `items`
- Mỗi `item`: `paper`, `review`, `mode`, `paper_year`, `queries`, `candidate_pool_top30`, `stats`

## Phase 4 (Metrics: CFI + CSP + Novelty)
Phase này tổng hợp metric cuối từ:
- `phase2/output/Master_Classification.jsonl`
- `phase3/output/Novelty_Assessment_Report.json`

### Chạy Phase4
```powershell
cd C:\Users\HOY9HC\Desktop\Code\Unified_pipeline\phase4
python run_phase4.py -v
```

Output mặc định:
- `phase4/output/Phase4_Metrics_Report.json`
- `phase4/output/Phase4_Metrics_Report.md`

Trong đó:
- `cfi`: consensus flaw weights + reviewer performance
- `csp`: `CPS` và `NSR` theo reviewer
- `novelty`: novelty risk score theo reviewer (dựa trên novelty claims + retrieved candidates từ Phase3)
- `summary_by_reviewer_type`: trung bình theo nhóm `Meta/Human/LLM`

## Test Phase1 với LLM khác (Azure OpenAI)
Thiết lập biến môi trường (AAD + deployment):

```powershell
$env:AZURE_TENANT_ID="<tenant-id>"
$env:AZURE_CLIENT_ID="<client-id>"
$env:AZURE_CLIENT_SECRET="<client-secret>"
$env:AZURE_ENDPOINT="https://<your-resource>.openai.azure.com"
$env:AZURE_CHAT_DEPLOYMENT="<deployment-name>"
$env:AZURE_API_VERSION="2024-10-21"
```

Chạy Phase1 bằng Azure backend:

```powershell
python phase1\run_phase1.py --paper-id 0A5o6dCKeK --llm-provider azure -v
```

## Ghi chú về LLM
- Nếu có `GEMINI_API_KEY` trong môi trường hoặc `.env`, pipeline sẽ gọi Gemini để atomize/classify/extract.
- Nếu chưa có key, pipeline chạy fallback heuristic để vẫn sinh output đúng schema.
- Phase1 hỗ trợ `--llm-provider azure` để bypass luồng Gemini khi proxy nội bộ gây lỗi.

## Schema chính
- Phase 1: giữ `parsed_sections -> raw_text + spacy_sentences + llm_atomic_arguments`
- Phase 2 master: `spacy_sentences_classification` + `llm_atomic_arguments_classification`
- Phase 2 novelty: `paper` + `review_novelty_extracted`
