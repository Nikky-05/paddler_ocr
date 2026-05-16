# Technical Pipeline Overview: Paddler OCR 🚀

This document provides a step-by-step walkthrough of the **Paddler OCR** processing pipeline, designed for technical interviews.

---

## 🏗️ 1. Project Architecture (File Roles)

The project is structured vertically from the **API Layer** to the **Extraction Layer**.

| File / Directory | Responsibility |
| :--- | :--- |
| **`main.py`** | **Entry Point.** FastAPI startup, OCR initialization, routing, and high-level orchestration. |
| **`extractors/`** | **Logic Layer.** Contains domain-specific rules (regex, row/column logic) for each ID document. |
| `extractors/kyc_report.py` | Handles "KYC Report" Aadhaar tables (different layout than raw cards). |
| `extractors/aadhaar.py` | Handles raw Front/Back Aadhaar cards from the UIDAI format. |
| `extractors/document_validator.py` | Ensures the image uploaded matches the requested type (e.g., stops PAN from being sent as Aadhaar). |
| `extractors/utils.py` | Shared helper functions like cleaning text, date normalization, and spatial matching. |

---

## 🌊 2. End-to-End Processing Pipeline

### **Step 1: System Startup (The `lifespan` Event)**
When the FastAPI server starts:
1.  **`lifespan` (main.py:781)**: Triggers the startup event.
2.  **`load_ocr_model()`**: Runs the OCR initialization in a separate thread (executor).
3.  **`_init_ocr()`**: Loads **PaddleOCR** with optimized "small text" parameters (`text_det_thresh=0.2`).
    *   *Interview Note*: "We use a lower threshold to capture faint or tiny text like Date of Birth on PVC cards."

### **Step 2: Receiving the Request (`POST /process`)**
When a client sends an image:
1.  **`process_document()`**: Reads the byte content of the file.
2.  **PDF-to-Image (main.py:562)**: If the input is a PDF, **PyMuPDF (fitz)** converts it into a high-resolution 300 DPI image for better OCR.

### **Step 3: Triple-Pass OCR (The Robustness Phase)**
Instead of running OCR once, the system uses multiple passes to ensure accuracy (**main.py:311**):
1.  **Pass 1 (Original)**: Standard OCR on the raw image.
2.  **Pass 2 (Preprocessed)**: Resizes the image and applies **Contrast Enhancement + Sharpening**.
3.  **Pass 3 (Binarized - PAN Only)**: Converts image to strict Black & White (Binarization) to separate text from complex backgrounds.
4.  **`merge_results()`**: All three passes are blended together. Duplicate lines are merged based on their spatial (Y-axis) coordinates.

### **Step 4: Smart Document Routing**
Before extracting, the system identifies the layout:
1.  **`is_kyc_report()`**: Detects if the image is a table-based "Digital KYC Report" instead of a raw card.
2.  **`validate_document_type()`**: Checks for keywords (e.g., "Income Tax" for PAN) to ensure the file is valid.

### **Step 5: Semantic Extraction (The "Extractors" Layer)**
The cleaned OCR text is passed to specific modules (e.g., `extractors/aadhaar.py`):
1.  **Regex Matching**: Finds patterns like `XXXX XXXX 1234` for Aadhaar or `ABCDE1234F` for PAN.
2.  **Spatial Logic**: If a label is "Name", the system looks at the **same line** or the **next line** to find the actual value.
3.  **Backside Cropping (main.py:730)**: For Aadhaar back-sides (which are bilingual), the system crops the left 52% of the image to isolate English text and avoid "garbage" characters from overlapping Hindi text.

### **Step 6: Face Verification & Address Splitting**
1.  **Face Detection**: Uses **MTCNN** to detect human faces in the document and returns a base64-encoded crop.
2.  **Address Splitting**: Uses `split_address` to break a long string into "Building", "Street", "Pincode", etc., for easier database insertion.

### **Step 7: The Uniform Response**
Finally, **`create_uniform_response()`** (main.py:172) gathers everything into a standard JSON schema containing:
*   **Doc Type**
*   **Extracted Values**
*   **Confidence Scores** (calculated from the OCR engine's output).

---

## 🏆 3. Summary for Interview
"My pipeline is designed for **maximum reliability** in real-world scenarios. We use a **Triple-Pass OCR strategy** (preprocessing, sharpening, and binarization) to handle low-light photos. We then use **Spatial Logic** and **Bilingual Isolation (cropping)** to extract clean data from complex layouts like Aadhaar and PAN cards, all wrapped in an asynchronous **FastAPI** wrapper for high performance."
