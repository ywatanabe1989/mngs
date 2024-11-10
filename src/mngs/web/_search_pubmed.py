#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-11 05:26:14 (ywatanabe)"
# File: ./mngs_repo/src/mngs/web/_search_pubmed.py

"""
1. Functionality:
   - Searches PubMed database for scientific articles
   - Retrieves detailed information about matched articles
   - Displays article metadata including title, authors, journal, year, and abstract
2. Input:
   - Search query string (e.g., "epilepsy prediction")
   - Optional parameters for batch size and result limit
3. Output:
   - Formatted article information displayed to stdout
   - BibTeX file with official citations
4. Prerequisites:
   - Internet connection
   - requests package
   - mngs package
"""

"""Imports"""
import argparse
import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Union

import aiohttp
import mngs
import requests

"""Functions & Classes"""


def search_pubmed(query: str, retmax: int = 10) -> Dict[str, Any]:
    """[Previous docstring remains the same]"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "y",
    }

    # print(f"Searching PubMed for: {query}")
    response = requests.get(search_url, params=params)
    if not response.ok:
        # print(f"Error: {response.status_code} - {response.text}")
        return {}
    return response.json()


# def search_pubmed(query: str, max_retries: int = 3, retry_delay: int = 5) -> Dict:
#     """Search PubMed with retries and proper API usage."""
#     base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#     params = {
#         "db": "pubmed",
#         "term": query,
#         "retmax": 100,
#         "retmode": "json",
#         "usehistory": "y",
#         "tool": "mngs",  # Add tool name
#         "email": "your.email@example.com"  # Add your email
#     }

#     for attempt in range(max_retries):
#         try:
#             response = requests.get(base_url, params=params, timeout=30)
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.RequestException as e:
#             if attempt == max_retries - 1:
#                 raise RuntimeError(f"Failed to connect to PubMed after {max_retries} attempts: {e}")
#             # print(f"Connection failed, retrying in {retry_delay} seconds...")
#             time.sleep(retry_delay)

#     return {}


def fetch_details(
    webenv: str, query_key: str, retstart: int = 0, retmax: int = 100
) -> Dict[str, Any]:
    """Fetches detailed information including abstracts for articles.

    Parameters
    ----------
    [Previous parameters remain the same]

    Returns
    -------
    Dict[str, Any]
        Dictionary containing article details and abstracts
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Fetch abstracts
    efetch_url = f"{base_url}efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "xml",
        "rettype": "abstract",
        "field": "abstract,mesh"  # Add MeSH terms
    }

    abstract_response = requests.get(efetch_url, params=efetch_params)

    # Fetch metadata
    fetch_url = f"{base_url}esummary.fcgi"
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json",
    }

    details_response = requests.get(fetch_url, params=params)

    if not all([abstract_response.ok, details_response.ok]):
        # print(f"Error fetching data")
        return {}

    return {
        "abstracts": abstract_response.text,
        "details": details_response.json(),
    }


def parse_abstract_xml(xml_text: str) -> Dict[str, str]:
    """Parses XML response to extract abstracts.

    Parameters
    ----------
    xml_text : str
        XML response from PubMed

    Returns
    -------
    Dict[str, str]
        Dictionary mapping PMIDs to abstracts
    """
    root = ET.fromstring(xml_text)
    results = {}

    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        abstract_element = article.find(".//Abstract/AbstractText")
        abstract = abstract_element.text if abstract_element is not None else ""

        # Get MeSH terms
        keywords = []
        mesh_terms = article.findall(".//MeshHeading/DescriptorName")
        keywords = [term.text for term in mesh_terms if term is not None]

        results[pmid] = (abstract, keywords)

    return results


def get_citation(pmid: str) -> str:
    """Gets official citation in BibTeX format.

    Parameters
    ----------
    pmid : str
        PubMed ID

    Returns
    -------
    str
        Official BibTeX citation
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    cite_url = f"{base_url}efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "bibtex",
        "retmode": "text",
    }

    response = requests.get(cite_url, params=params)
    return response.text if response.ok else ""


def save_bibtex(
    papers: Dict[str, Any], abstracts: Dict[str, str], output_file: str
) -> None:
    """Saves paper metadata as BibTeX file with abstracts.

    Parameters
    ----------
    papers : Dict[str, Any]
        Dictionary of paper metadata
    abstracts : Dict[str, str]
        Dictionary of PMIDs to abstracts
    output_file : str
        Output file path
    """
    with open(output_file, "w", encoding="utf-8") as bibtex_file:
        for pmid, paper in papers.items():
            if pmid == "uids":
                continue

            citation = get_citation(pmid)
            if citation:
                bibtex_file.write(citation)
            else:
                # Fallback to our formatted entry
                bibtex_entry = format_bibtex(
                    paper, pmid, abstracts.get(pmid, "")
                )
                bibtex_file.write(bibtex_entry + "\n")


# def format_bibtex(paper: Dict[str, Any], pmid: str, abstract: str = "") -> str:
#     """Formats paper metadata as BibTeX entry.

#     Parameters
#     ----------
#     paper : Dict[str, Any]
#         Paper metadata dictionary
#     pmid : str
#         PubMed ID
#     abstract : str, optional
#         Paper abstract

#     Returns
#     -------
#     str
#         Formatted BibTeX entry
#     """
#     authors = paper.get("authors", [{"name": "Unknown"}])
#     author_names = " and ".join(author["name"] for author in authors)
#     year = paper.get("pubdate", "").split()[0]
#     title = paper.get("title", "No Title")
#     journal = paper.get("source", "Unknown Journal")

#     entry = f"""@article{{pmid{pmid},
#     author = {{{author_names}}},
#     title = {{{title}}},
#     journal = {{{journal}}},
#     year = {{{year}}},
#     pmid = {{{pmid}}},
#     abstract = {{{abstract}}}
# }}
# """
#     return entry
# def format_bibtex(paper: Dict[str, Any], pmid: str, abstract: str = "") -> str:
#     authors = paper.get("authors", [{"name": "Unknown"}])
#     author_names = " and ".join(author["name"] for author in authors)
#     year = paper.get("pubdate", "").split()[0]
#     title = paper.get("title", "No Title")
#     journal = paper.get("source", "Unknown Journal")

#     # Create citation key: FirstAuthorLastName_Year_FirstTitleWord
#     first_author = authors[0]["name"].split()[-1].lower()  # Get last name of first author
#     first_title_word = title.split()[0].lower()
#     citation_key = f"{first_author}_{year}_{first_title_word}"

#     entry = f"""@article{{{citation_key},
#     author = {{{author_names}}},
#     title = {{{title}}},
#     journal = {{{journal}}},
#     year = {{{year}}},
#     pmid = {{{pmid}}},
#     abstract = {{{abstract}}}
# }}
# """
#     return entry


def format_bibtex(paper: Dict[str, Any], pmid: str, abstract_data) -> str:
    abstract, keywords = abstract_data if isinstance(abstract_data, tuple) else (abstract_data, [])
    authors = paper.get("authors", [{"name": "Unknown"}])
    author_names = " and ".join(author["name"] for author in authors)
    year = paper.get("pubdate", "").split()[0]
    title = paper.get("title", "No Title")
    journal = paper.get("source", "Unknown Journal")

    # Better name formatting
    first_author = authors[0]["name"]
    first_name = first_author.split()[0]
    last_name = first_author.split()[-1]  # Take last part as surname
    # Remove special characters and spaces, convert to lowercase
    clean_first_name = "".join(c for c in first_name if c.isalnum())
    clean_last_name = "".join(c for c in last_name if c.isalnum())

    # Clean first word of title
    first_title_word = "".join(
        c.lower() for c in title.split()[0] if c.isalnum()
    )
    second_title_word = "".join(
        c.lower() for c in title.split()[1] if c.isalnum()
    )


    citation_key = f"{clean_first_name}.{clean_last_name}_{year}_{first_title_word}_{second_title_word}"

    entry = f"""@article{{{citation_key},
    author = {{{author_names}}},
    title = {{{title}}},
    journal = {{{journal}}},
    year = {{{year}}},
    pmid = {{{pmid}}},
    keywords = {{{", ".join(keywords)}}},
    abstract = {{{abstract}}}
}}
"""
    return entry


# async def fetch_async(
#     session: aiohttp.ClientSession, url: str, params: Dict
# ) -> Dict:
#     """Asynchronous fetch helper."""
#     async with session.get(url, params=params) as response:
#         if response.status == 200:
#             if "json" in params.get("retmode", ""):
#                 return await response.json()
#             return await response.text()
#         return {}


async def fetch_async(
    session: aiohttp.ClientSession, url: str, params: Dict
) -> Union[Dict, str]:
    """Asynchronous fetch helper."""
    async with session.get(url, params=params) as response:
        if response.status == 200:
            if params.get("retmode") == "xml":
                return await response.text()
            elif params.get("retmode") == "json":
                return await response.json()
            return await response.text()
        return {}


async def batch_fetch_details(
    pmids: List[str], batch_size: int = 20
) -> List[Dict]:
    """Fetches details for multiple PMIDs concurrently.

    Parameters
    ----------
    pmids : List[str]
        List of PubMed IDs
    batch_size : int, optional
        Size of each batch for concurrent requests

    Returns
    -------
    List[Dict]
        List of response data
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i : i + batch_size]

            # Fetch both details and citations concurrently
            efetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "xml",
                "rettype": "abstract",
            }

            esummary_params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "json",
            }

            tasks.append(
                fetch_async(session, f"{base_url}efetch.fcgi", efetch_params)
            )
            tasks.append(
                fetch_async(
                    session, f"{base_url}esummary.fcgi", esummary_params
                )
            )

        results = await asyncio.gather(*tasks)
        return results


def main(args: argparse.Namespace, n_entries: int = 10) -> int:
    query = args.query or "epilepsy prediction"
    # print(f"Using query: {query}")

    search_results = search_pubmed(query)
    if not search_results:
        # print("No results found or error occurred")
        return 1

    pmids = search_results["esearchresult"]["idlist"]
    count = len(pmids)
    # print(f"Found {count:,} results")

    output_file = f"pubmed_{query.replace(' ', '_')}.bib"
    # print(f"Saving results to: {output_file}")

    # Process in larger batches asynchronously
    results = asyncio.run(batch_fetch_details(pmids[:n_entries]))
    # here, results seems long string

    # Process results and save
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(0, len(results), 2):
            xml_response = results[i]
            json_response = results[i + 1]

            if isinstance(xml_response, str):
                abstracts = parse_abstract_xml(xml_response)
                if (
                    isinstance(json_response, dict)
                    and "result" in json_response
                ):
                    details = json_response["result"]
                    save_bibtex(details, abstracts, output_file)

    # Process results and save
    temp_bibtex = []  # Store entries temporarily
    for i in range(0, len(results), 2):
        xml_response = results[i]
        json_response = results[i + 1]

        if isinstance(xml_response, str):
            abstracts = parse_abstract_xml(xml_response)
            if isinstance(json_response, dict) and "result" in json_response:
                details = json_response["result"]
                for pmid in details:
                    if pmid != "uids":
                        citation = get_citation(pmid)
                        if citation:
                            temp_bibtex.append(citation)
                        else:
                            entry = format_bibtex(
                                details[pmid], pmid, abstracts.get(pmid, "")
                            )
                            temp_bibtex.append(entry)

    # Write all entries at once
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(temp_bibtex))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PubMed article search and retrieval tool"
    )
    parser.add_argument(
        "--bibtex",
        "-b",
        action="store_true",
        help="Save results as BibTeX file",
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help='Search query (default: "epilepsy prediction")',
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    global CONFIG
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        verbose=False,
    )

    args = parse_args()
    exit_status = main(args)

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
