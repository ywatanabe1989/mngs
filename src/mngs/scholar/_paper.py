#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-12-06 10:00:00"
# Author: Claude
# Filename: _paper.py

"""
Paper class for representing scientific papers.
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import json


class Paper:
    """Represents a scientific paper with metadata and content."""
    
    def __init__(
        self,
        title: str,
        authors: List[str],
        abstract: str,
        source: str,  # 'pubmed', 'arxiv', 'local', etc.
        year: Optional[int] = None,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        journal: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        pdf_path: Optional[Path] = None,
        embedding: Optional[Any] = None,  # numpy array or tensor
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a Paper object.
        
        Parameters
        ----------
        title : str
            Paper title
        authors : List[str]
            List of author names
        abstract : str
            Paper abstract
        source : str
            Source of the paper (pubmed, arxiv, local, etc.)
        year : int, optional
            Publication year
        doi : str, optional
            Digital Object Identifier
        pmid : str, optional
            PubMed ID
        arxiv_id : str, optional
            arXiv identifier
        journal : str, optional
            Journal name
        keywords : List[str], optional
            Keywords/tags
        pdf_path : Path, optional
            Path to local PDF file
        embedding : Any, optional
            Vector embedding of the paper
        metadata : Dict[str, Any], optional
            Additional metadata
        """
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.source = source
        self.year = year
        self.doi = doi
        self.pmid = pmid
        self.arxiv_id = arxiv_id
        self.journal = journal
        self.keywords = keywords or []
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.embedding = embedding
        self.metadata = metadata or {}
        self.retrieved_at = datetime.now()
    
    def __repr__(self) -> str:
        """String representation of the paper."""
        return f"Paper(title='{self.title[:50]}...', authors={len(self.authors)}, source='{self.source}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} authors)"
        
        parts = [
            f"Title: {self.title}",
            f"Authors: {authors_str}",
            f"Year: {self.year or 'Unknown'}",
            f"Source: {self.source}",
        ]
        
        if self.journal:
            parts.append(f"Journal: {self.journal}")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        if self.pmid:
            parts.append(f"PMID: {self.pmid}")
        if self.arxiv_id:
            parts.append(f"arXiv: {self.arxiv_id}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary representation."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "source": self.source,
            "year": self.year,
            "doi": self.doi,
            "pmid": self.pmid,
            "arxiv_id": self.arxiv_id,
            "journal": self.journal,
            "keywords": self.keywords,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "has_embedding": self.embedding is not None,
            "metadata": self.metadata,
            "retrieved_at": self.retrieved_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        """Create Paper from dictionary representation."""
        # Remove computed fields
        data = data.copy()
        data.pop("has_embedding", None)
        data.pop("retrieved_at", None)
        
        # Convert pdf_path back to Path if present
        if data.get("pdf_path"):
            data["pdf_path"] = Path(data["pdf_path"])
        
        return cls(**data)
    
    def to_bibtex(self) -> str:
        """Generate BibTeX entry for the paper."""
        # Generate citation key
        first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        year = self.year or "0000"
        title_word = self.title.split()[0].lower() if self.title else "paper"
        cite_key = f"{first_author}{year}{title_word}"
        
        # Determine entry type
        if self.arxiv_id:
            entry_type = "@misc"
        else:
            entry_type = "@article"
        
        # Build BibTeX entry
        lines = [f"{entry_type}{{{cite_key},"]
        lines.append(f'  title = "{{{self.title}}}",')
        
        if self.authors:
            authors_str = " and ".join(self.authors)
            lines.append(f'  author = "{{{authors_str}}}",')
        
        if self.year:
            lines.append(f'  year = "{{{self.year}}}",')
        
        if self.journal:
            lines.append(f'  journal = "{{{self.journal}}}",')
        
        if self.doi:
            lines.append(f'  doi = "{{{self.doi}}}",')
        
        if self.arxiv_id:
            lines.append(f'  eprint = "{{{self.arxiv_id}}}",')
            lines.append('  archivePrefix = "{arXiv}",')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def get_identifier(self) -> str:
        """Get a unique identifier for the paper."""
        if self.doi:
            return f"doi:{self.doi}"
        elif self.pmid:
            return f"pmid:{self.pmid}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        else:
            # Create a hash from title and authors
            import hashlib
            content = f"{self.title}_{';'.join(self.authors)}"
            return f"hash:{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def has_pdf(self) -> bool:
        """Check if paper has an associated PDF file."""
        return self.pdf_path is not None and self.pdf_path.exists()
    
    def similarity_score(self, other: "Paper") -> float:
        """Calculate similarity score with another paper (0-1)."""
        score = 0.0
        weights = {"title": 0.4, "authors": 0.3, "abstract": 0.3}
        
        # Title similarity (simple word overlap)
        title_words1 = set(self.title.lower().split())
        title_words2 = set(other.title.lower().split())
        if title_words1 and title_words2:
            title_overlap = len(title_words1 & title_words2) / len(title_words1 | title_words2)
            score += weights["title"] * title_overlap
        
        # Author similarity
        authors1 = set(self.authors)
        authors2 = set(other.authors)
        if authors1 and authors2:
            author_overlap = len(authors1 & authors2) / len(authors1 | authors2)
            score += weights["authors"] * author_overlap
        
        # Abstract similarity (simple word overlap)
        abstract_words1 = set(self.abstract.lower().split())
        abstract_words2 = set(other.abstract.lower().split())
        if abstract_words1 and abstract_words2:
            abstract_overlap = len(abstract_words1 & abstract_words2) / len(abstract_words1 | abstract_words2)
            score += weights["abstract"] * abstract_overlap
        
        return score


# Example usage
if __name__ == "__main__":
    paper = Paper(
        title="Deep Learning for Scientific Discovery",
        authors=["John Doe", "Jane Smith", "Bob Johnson"],
        abstract="This paper explores the application of deep learning...",
        source="arxiv",
        year=2024,
        arxiv_id="2401.12345",
        keywords=["deep learning", "scientific computing", "AI"],
    )
    
    print(paper)
    print("\nBibTeX:")
    print(paper.to_bibtex())
    print("\nIdentifier:", paper.get_identifier())