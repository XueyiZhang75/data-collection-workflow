# Hondius Hantavirus Outbreak Data Repository 2026

This repository contains a curated epidemiological line list of cases associated with the **ongoing 2026 Hantavirus outbreak**. The outbreak has been linked to the **Andes virus (ANDV)** strain.

The primary line list data is stored in `data/linelist/2026_hantavirus.csv. We are also curating data related to exposures, sequences, and news articles. See Data section below. 

## Citation
Please cite: "Global.health: Hondius hantavirus 2026 outbreak data [YYYY-MM-DD]" and also acknowledge/cite the data sources listed. 

## Contributing
Please feel free to contribute to this tracking effort:

* **Pull Requests:** Submit a PR for data corrections or new entries.
* **Data Integrity:** Ensure all new entries are supported by a reputable source (WHO, CDC, or national health ministries).
* **Privacy Guardrail:** **Do not upload PII (Personally Identifiable Information).**

## Data
We are curating four kinds of data. The first is a line list of suspected/confirmed cases. This line list pulls from publicly available sources and provides key metadata for each case. The second is a list of known exposures. These exposures were curated using an agentic workflow that we call episource. The third are sequencing data from the current outbreak. Currently, this is just a single sequence provided by the Swiss National Reference Center for Emerging Viral Infections, Geneva University Hospitals and the Institute of Medical Virology, University of Zurich. Finally, a set of news article that were sourced from https://www.mediacloud.org/ and then extracted and structured using the episource agentic workflow. 

1. Line list - data/linelist/2026_hantavirus.csv
2. Exposure list - data/exposures/2026_hantavirus_exposures.csv
3. Sequences - data/sequences/ANDV-Switzerland-Hu-3337-2026.fasta.gz AND readme.txt (important context around the sequence and citation). 
4. News articles - data/news sources/mc-onlinenews-mediacloud-20260507155422-content.csv (list of news articles sourced from Media Cloud) AND hantavirus_articles.json (raw text from news articles) AND hantavirus_results.json (structured data). 

## License
This data is made available under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## Additional license, warranty, and copyright information
We provide a license for our code (see LICENSE) and do not claim ownership, nor the right to license, the data we have obtained nor any third-party software tools/code used in our analyses.  Please cite the appropriate agency, paper, and/or individual in publications and/or derivatives using these data, contact them regarding the legal use of these data, and remember to pass-forward any existing license/warranty/copyright information.  THE DATA AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA AND/OR SOFTWARE OR THE USE OR OTHER DEALINGS IN THE DATA AND/OR SOFTWARE.