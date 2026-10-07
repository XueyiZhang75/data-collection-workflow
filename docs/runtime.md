# Runtime setup

Evidence mode requires a local Chromium browser, PDFium (installed with the Python dependencies), and Tesseract 5 with `eng`, `osd`, `fra`, `spa`, `por`, and `chi_sim` language data. It checks the browser, text PDF parsing, and scanned-PDF OCR locally before provider requests. Standard-mode runs using local content fixtures do not need browser/OCR setup.

## Chromium

Install the browser for the Playwright version in the active environment:

```bash
python -m playwright install chromium
```

Linux systems may also need Playwright's system dependencies. See the [official browser installation instructions](https://playwright.dev/python/docs/browsers). A custom executable can be supplied through `CHROMIUM_EXECUTABLE`; otherwise Playwright locates its installed browser.

## Tesseract

Install Tesseract 5 and the language files listed above. Set `TESSERACT_EXECUTABLE` to the executable and `TESSDATA_DIR` to the language-data directory, or provide them in a local `.runtime/acquisition-paths.json` file:

```json
{
  "tesseract_executable": "/path/to/tesseract",
  "tessdata_dir": "/path/to/tessdata"
}
```

The paths above are placeholders. On Windows, use a full executable path such as `C:/Tools/Tesseract/tesseract.exe` with the matching language-data directory. Some Windows builds cannot handle Unicode paths passed to OCR. For those installations, add `tesseract_working_directory`, `tessdata_argument`, and an `ocr_staging_directory` with an ASCII-compatible path. The executable, data files, fonts, and these paths belong to each user's machine.

Optional settings include `chromium_executable`, `playwright_browsers_path`, and `chromium_mode`. Equivalent acquisition settings can be supplied under `universal.acquisition` in a task configuration. The `.runtime/` directory is ignored by Git.

The local OCR readiness check renders a synthetic scanned page and needs a TrueType font. It tries `arial.ttf` on Windows and `DejaVuSans.ttf` on other systems. If readiness fails with `cannot open resource`, install the corresponding font or add `"preflight_font": "/path/to/your/font.ttf"` to `.runtime/acquisition-paths.json`, using a real font file on your machine.

Browser and OCR tests use local synthetic pages and PDFs. A passing local readiness check does not check remote website access or a provider account's available credit.
