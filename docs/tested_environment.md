# Tested installation environment

The core command-line workflow and software tests are verified with **64-bit CPython 3.12.2 on Windows 11**. [The version constraints](../constraints/python312-windows.txt) pin the direct and transitive Python dependencies used for this environment. They cover collection, evidence checks, the HTML report, and tests. Optional dashboard, Langflow, and Studio dependencies are separate.

To use these versions, follow README Step 1 with Python 3.12.2 and replace its `pip install` command with:

```powershell
python -m pip install -c constraints/python312-windows.txt setuptools wheel
python -m pip install --no-build-isolation -c constraints/python312-windows.txt -e .
python -m pip check
```

Run these commands inside the activated virtual environment, from the source checkout. A successful dependency check prints `No broken requirements found.` The constraints file does not install the project by itself; `-e .` installs the project and its required dependencies.

The first command installs the verified build tools (setuptools 80.9.0 and wheel 0.47.0). `--no-build-isolation` uses those installed tools when building the editable project.

**Use a short ASCII checkout path on Windows**, such as `C:/Projects/data-collection-workflow`. On Python 3.12, editable installation can fail to encode a source path containing Chinese or other characters outside the Windows locale; move the checkout to an ASCII path and recreate the virtual environment there.

The general installation in the README remains available for other supported Python versions and platforms. This recorded version set was tested on the Windows/Python combination above. Installing optional interfaces can introduce additional dependencies; use a separate virtual environment when their version requirements differ.

Continue with README Steps 2 and 3 to configure credentials and install Chromium and Tesseract. Chromium must match the installed Playwright version; install it using `python -m playwright install chromium` in this environment. The OCR requirement is Tesseract 5 with `eng`, `osd`, `fra`, `spa`, `por`, and `chi_sim` data. Python dependency constraints do not install these external tools. See [Runtime setup](runtime.md).

To check the installed command and then run the software tests:

```powershell
data-collection-workflow --help
python -m pip install --no-build-isolation -c constraints/python312-windows.txt -e ".[test]"
python -m pytest
```

The dependency check and software tests verify installation consistency and program behavior. A live collection still depends on the configured accounts, the selected model, source availability, and the requested evidence.
