"""
Document processor for the research analyst system.
Fetches, cleans, and chunks web documents.
"""

import re
from typing import List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import trafilatura

from research_analyst.core.models import Document, DocumentChunk
from research_analyst.core.exceptions import DocumentFetchError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import (
    generate_id,
    clean_text,
    chunk_list,
    estimate_token_count
)


logger = get_logger()


class DocumentProcessor:
    """Process and chunk web documents."""
    
    def __init__(self):
        """Initialize document processor."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Chunking parameters
        self.chunk_size = 1500  # characters
        self.chunk_overlap = 300  # characters
        self.max_chunks_per_doc = 20
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_document(self, document: Document) -> Document:
        """
        Fetch full content for a document.
        
        Args:
            document: Document with URL
            
        Returns:
            Document with fetched content
            
        Raises:
            DocumentFetchError: If fetch fails
        """
        url = str(document.url)
        
        self.logger.debug(
            "Fetching document",
            url=url,
            doc_id=document.doc_id
        )
        
        try:
            # Try trafilatura first (better for articles)
            content = self._fetch_with_trafilatura(url)
            
            if not content or len(content) < 100:
                # Fallback to BeautifulSoup
                content = self._fetch_with_beautifulsoup(url)
            
            if not content:
                raise DocumentFetchError(
                    f"Failed to extract content from {url}",
                    details={"url": url, "doc_id": document.doc_id}
                )
            
            # Update document
            document.content = content
            
            # Extract metadata if possible
            metadata = self._extract_metadata(url, content)
            document.metadata.update(metadata)
            
            if metadata.get('author'):
                document.author = metadata['author']
            if metadata.get('published_date'):
                document.published_date = metadata['published_date']
            
            self.logger.info(
                "Document fetched",
                doc_id=document.doc_id,
                content_length=len(content),
                url=url
            )
            
            return document
            
        except requests.RequestException as e:
            raise DocumentFetchError(
                f"Network error fetching document: {str(e)}",
                details={"url": url, "error": str(e)}
            )
        except Exception as e:
            raise DocumentFetchError(
                f"Error processing document: {str(e)}",
                details={"url": url, "error": str(e)}
            )
    
    def _fetch_with_trafilatura(self, url: str) -> Optional[str]:
        """
        Fetch content using trafilatura.
        
        Args:
            url: Document URL
            
        Returns:
            Extracted content or None
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False
                )
                return content
        except Exception as e:
            self.logger.debug(
                "Trafilatura extraction failed",
                url=url,
                error=str(e)
            )
        return None
    
    def _fetch_with_beautifulsoup(self, url: str) -> Optional[str]:
        """
        Fetch content using BeautifulSoup.
        
        Args:
            url: Document URL
            
        Returns:
            Extracted content or None
        """
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.settings.request_timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.debug(
                "BeautifulSoup extraction failed",
                url=url,
                error=str(e)
            )
        return None
    
    def _extract_metadata(self, url: str, content: str) -> dict:
        """
        Extract metadata from content.
        
        Args:
            url: Document URL
            content: Document content
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Try to extract author (simple heuristic)
        author_patterns = [
            r'[Bb]y\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'[Aa]uthor:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content[:500])
            if match:
                metadata['author'] = match.group(1)
                break
        
        # Try to extract date (simple heuristic)
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content[:500])
            if match:
                try:
                    import dateparser
                    parsed_date = dateparser.parse(match.group(1))
                    if parsed_date:
                        metadata['published_date'] = parsed_date
                        break
                except:
                    pass
        
        return metadata
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk document content for embedding.
        
        Args:
            document: Document with content
            
        Returns:
            List of DocumentChunk objects
        """
        if not document.content:
            self.logger.warning(
                "Cannot chunk document without content",
                doc_id=document.doc_id
            )
            return []
        
        content = document.content
        
        # Clean content
        content = clean_text(content)
        
        # Split into chunks
        chunks = self._split_into_chunks(content)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=generate_id("chunk"),
                doc_id=document.doc_id,
                text=chunk_text,
                chunk_index=i,
                metadata={
                    'source_url': str(document.url),
                    'source_title': document.title,
                    'chunk_size': len(chunk_text),
                    'estimated_tokens': estimate_token_count(chunk_text)
                }
            )
            document_chunks.append(chunk)
        
        # Limit to max chunks
        if len(document_chunks) > self.max_chunks_per_doc:
            self.logger.warning(
                "Document has too many chunks, truncating",
                doc_id=document.doc_id,
                total_chunks=len(document_chunks),
                max_chunks=self.max_chunks_per_doc
            )
            document_chunks = document_chunks[:self.max_chunks_per_doc]
        
        self.logger.info(
            "Document chunked",
            doc_id=document.doc_id,
            num_chunks=len(document_chunks)
        )
        
        return document_chunks
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Keep overlap from previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + paragraph
                else:
                    # Paragraph itself is too long, split it
                    if len(paragraph) > self.chunk_size:
                        split_para = self._split_long_paragraph(paragraph)
                        chunks.extend(split_para[:-1])
                        current_chunk = split_para[-1] if split_para else ""
                    else:
                        current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """
        Split a paragraph that's too long.
        
        Args:
            paragraph: Long paragraph
            
        Returns:
            List of chunks
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap
                    overlap = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap + " " + sentence
                else:
                    # Single sentence too long, hard split
                    chunks.append(sentence[:self.chunk_size])
                    current_chunk = sentence[self.chunk_size:]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_documents(
        self,
        documents: List[Document],
        fetch_content: bool = True,
        create_chunks: bool = True
    ) -> tuple[List[Document], List[DocumentChunk]]:
        """
        Process multiple documents.
        
        Args:
            documents: List of documents
            fetch_content: Whether to fetch full content
            create_chunks: Whether to create chunks
            
        Returns:
            Tuple of (processed_documents, all_chunks)
        """
        self.logger.info(
            "Processing documents",
            num_documents=len(documents),
            fetch_content=fetch_content,
            create_chunks=create_chunks
        )
        
        processed_documents = []
        all_chunks = []
        
        for doc in documents:
            try:
                # Fetch content if requested
                if fetch_content and len(doc.content or "") < 500:
                    doc = self.fetch_document(doc)
                
                processed_documents.append(doc)
                
                # Create chunks if requested
                if create_chunks and doc.content:
                    chunks = self.chunk_document(doc)
                    all_chunks.extend(chunks)
                    
            except DocumentFetchError as e:
                self.logger.warning(
                    "Failed to process document",
                    doc_id=doc.doc_id,
                    url=str(doc.url),
                    error=str(e)
                )
                # Continue with other documents
                continue
        
        self.logger.info(
            "Document processing complete",
            processed=len(processed_documents),
            total_chunks=len(all_chunks)
        )
        
        return processed_documents, all_chunks
    
    def extract_key_sentences(self, document: Document, num_sentences: int = 3) -> List[str]:
        """
        Extract key sentences from document.
        
        Args:
            document: Document to process
            num_sentences: Number of sentences to extract
            
        Returns:
            List of key sentences
        """
        if not document.content:
            return []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', document.content)
        
        # Filter short sentences
        sentences = [s for s in sentences if len(s) > 50]
        
        # For now, return first N sentences (can be improved with scoring)
        return sentences[:num_sentences]