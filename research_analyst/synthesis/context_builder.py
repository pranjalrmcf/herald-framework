"""
Context builder for the research analyst system.
Structures evidence into claims, sources, and counter-arguments for synthesis.
"""

import json
from typing import List, Dict, Optional
from collections import defaultdict

from research_analyst.core.models import (
    Document,
    RankedDocument,
    Subgraph,
    Claim,
    Evidence,
    Relationship
)
from research_analyst.core.exceptions import ContextBuildingError, LLMError
from research_analyst.config import get_settings, prompts
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_id
from research_analyst.utils.llm_client import get_llm_client


logger = get_logger()


class ContextBuilder:
    """Build structured context from documents and graph data."""
    
    def __init__(self):
        """Initialize context builder."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Use unified LLM client (Groq)
        self.llm_client = get_llm_client()


    def build_simple_context(
        self,
        ranked_documents: List[RankedDocument],
    ) -> Evidence:
        """
        Fast context building for FAST path queries.
        Uses document snippets directly — no LLM claim extraction call.
        Saves 5-8 seconds vs full build_context().
        """
        from research_analyst.utils.helpers import generate_id
        from research_analyst.core.models import Claim

        documents = [rd.document for rd in ranked_documents[:5]]

        claims = []
        for rd in ranked_documents[:5]:
            doc = rd.document
            snippet = (
                doc.snippet
                or (doc.content or "")[:300]
            ).strip()

            if snippet:
                claims.append(Claim(
                    claim_id=generate_id("claim"),
                    text=snippet,
                    supporting_sources=[doc.doc_id],
                    confidence=round(rd.final_score, 3),
                    is_controversial=False,
                ))

        self.logger.info(
            "Simple context built (fast path)",
            num_claims=len(claims),
            num_documents=len(documents),
        )

        return Evidence(
            evidence_id=generate_id("ev_fast"),
            claims=claims,
            supporting_documents=documents,
            counter_arguments=[],
            relationship_chains=None,
            summary="Fast path evidence from document snippets.",
        )
    
    def build_context(
        self,
        ranked_documents: List[RankedDocument],
        subgraph: Optional[Subgraph] = None
    ) -> Evidence:
        """
        Build structured evidence from documents and graph.
        
        Args:
            ranked_documents: Ranked documents from retrieval
            subgraph: Optional knowledge graph subgraph
            
        Returns:
            Evidence object with structured claims and sources
        """
        self.logger.info(
            "Building context",
            num_documents=len(ranked_documents),
            has_subgraph=subgraph is not None
        )
        
        try:
            # Extract documents
            documents = [rd.document for rd in ranked_documents]
            
            # Extract claims from documents
            claims = self._extract_claims_from_documents(documents)
            
            # Identify counter-arguments
            counter_arguments = self._identify_counter_arguments(claims)
            
            # Extract relationship chains if graph available
            relationship_chains = None
            if subgraph and subgraph.relationships:
                relationship_chains = self._extract_relationship_chains(subgraph)
            
            # Build evidence object
            evidence = Evidence(
                evidence_id=generate_id("evidence"),
                claims=claims,
                supporting_documents=documents,
                counter_arguments=counter_arguments,
                relationship_chains=relationship_chains,
                summary=self._generate_summary(claims, documents)
            )
            
            self.logger.info(
                "Context built",
                num_claims=len(claims),
                num_counter_arguments=len(counter_arguments),
                num_relationship_chains=len(relationship_chains) if relationship_chains else 0
            )
            
            return evidence
            
        except Exception as e:
            self.logger.error(
                "Context building failed",
                error=str(e)
            )
            raise ContextBuildingError(
                f"Failed to build context: {str(e)}",
                details={"num_documents": len(ranked_documents)}
            )
    
    def _extract_claims_from_documents(
        self,
        documents: List[Document]
    ) -> List[Claim]:
        """
        Extract factual claims from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of Claim objects
        """
        all_claims = []
        
        # Process top documents (limit to avoid token costs)
        for doc in documents[:5]:  # Top 5 documents
            try:
                if self.settings.mock_llm_calls:
                    doc_claims = self._mock_extract_claims(doc)
                else:
                    doc_claims = self._llm_extract_claims(doc)
                
                all_claims.extend(doc_claims)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to extract claims from document",
                    doc_id=doc.doc_id,
                    error=str(e)
                )
                continue
        
        # Deduplicate similar claims
        deduplicated_claims = self._deduplicate_claims(all_claims)
        
        return deduplicated_claims
    
    def _llm_extract_claims(self, document: Document) -> List[Claim]:
        """
        Extract claims using LLM.
        
        Args:
            document: Document to process
            
        Returns:
            List of claims
        """
        # Use snippet or first part of content
        text = document.snippet or document.content[:2000]
        
        # Format prompt
        prompt = prompts.format_prompt(
            prompts.CLAIM_EXTRACTION,
            text=text,
            source_url=str(document.url)
        )
        
        try:
            # Use unified LLM client
            response_text = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a claim extraction expert. Always respond with valid JSON.",
                max_tokens=1000,
                temperature=0.1,
                json_mode=True
            )
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Convert to Claim objects
            claims = []
            for claim_data in result.get('claims', []):
                claim = Claim(
                    claim_id=generate_id("claim"),
                    text=claim_data['claim'],
                    supporting_sources=[document.doc_id],
                    confidence=claim_data.get('confidence', 0.7),
                    is_controversial=claim_data.get('is_controversial', False)
                )
                claims.append(claim)
            
            return claims
            
        except Exception as e:
            self.logger.warning(
                "LLM claim extraction failed",
                doc_id=document.doc_id,
                error=str(e)
            )
            return []
    
    def _mock_extract_claims(self, document: Document) -> List[Claim]:
        """
        Mock claim extraction for testing.
        
        Args:
            document: Document to process
            
        Returns:
            List of mock claims
        """
        # Simple heuristic: split by sentences and treat as claims
        text = document.snippet or document.content[:500]
        sentences = text.split('. ')
        
        claims = []
        for i, sentence in enumerate(sentences[:3]):  # Max 3 claims
            if len(sentence) > 20:  # Skip very short sentences
                claim = Claim(
                    claim_id=generate_id("claim"),
                    text=sentence.strip(),
                    supporting_sources=[document.doc_id],
                    confidence=0.6,
                    is_controversial=False
                )
                claims.append(claim)
        
        return claims
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Deduplicate similar claims.
        
        Args:
            claims: List of claims
            
        Returns:
            Deduplicated claims
        """
        # Simple deduplication by normalized text
        claim_map = {}
        
        for claim in claims:
            # Normalize text
            normalized = claim.text.lower().strip()
            
            if normalized in claim_map:
                # Merge sources
                existing = claim_map[normalized]
                existing.supporting_sources.extend(claim.supporting_sources)
                existing.supporting_sources = list(set(existing.supporting_sources))
                
                # Update confidence (average)
                existing.confidence = (existing.confidence + claim.confidence) / 2
                
                # Mark as controversial if any version is
                if claim.is_controversial:
                    existing.is_controversial = True
            else:
                claim_map[normalized] = claim
        
        return list(claim_map.values())
    
    def _identify_counter_arguments(self, claims: List[Claim]) -> List[str]:
        """
        Identify contradictions and counter-arguments among claims.
        
        Args:
            claims: List of claims
            
        Returns:
            List of counter-argument strings
        """
        if not claims or self.settings.mock_llm_calls:
            return []
        
        # Group claims by topic (simplified)
        claim_texts = [c.text for c in claims]
        
        # Format prompt
        prompt = prompts.format_prompt(
            prompts.COUNTER_ARGUMENT_DETECTION,
            claims="\n".join(f"{i+1}. {c}" for i, c in enumerate(claim_texts))
        )
        
        try:
            # Use unified LLM client
            response_text = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are an analytical expert. Always respond with valid JSON.",
                max_tokens=800,
                temperature=0.1,
                json_mode=True
            )
            
            # Parse result
            result = json.loads(response_text)
            
            # Extract counter-arguments
            counter_args = []
            for contradiction in result.get('contradictions', []):
                counter_args.append(contradiction.get('explanation', ''))
            
            counter_args.extend(result.get('nuances', []))
            
            return counter_args
            
        except Exception as e:
            self.logger.warning(
                "Counter-argument detection failed",
                error=str(e)
            )
            return []
    
    def _extract_relationship_chains(
        self,
        subgraph: Subgraph
    ) -> List[List[Relationship]]:
        """
        Extract meaningful relationship chains from graph.
        
        Args:
            subgraph: Knowledge graph subgraph
            
        Returns:
            List of relationship chains
        """
        # Group relationships by subject
        subject_map = defaultdict(list)
        for rel in subgraph.relationships:
            subject_map[rel.subject].append(rel)
        
        # Extract chains (simple 2-hop chains for now)
        chains = []
        
        for subject, rels in subject_map.items():
            for rel1 in rels:
                # Find relationships starting from rel1's object
                second_hop = subject_map.get(rel1.object, [])
                for rel2 in second_hop:
                    chains.append([rel1, rel2])
        
        # Sort by combined confidence
        chains.sort(
            key=lambda c: sum(r.confidence for r in c) / len(c),
            reverse=True
        )
        
        # Return top chains
        return chains[:10]
    
    def _generate_summary(
        self,
        claims: List[Claim],
        documents: List[Document]
    ) -> str:
        """
        Generate a brief summary of the evidence.
        
        Args:
            claims: Extracted claims
            documents: Source documents
            
        Returns:
            Summary string
        """
        lines = []
        
        lines.append(f"Evidence from {len(documents)} sources:")
        lines.append(f"- {len(claims)} key claims extracted")
        
        # Top 3 claims
        top_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)[:3]
        for i, claim in enumerate(top_claims, 1):
            lines.append(f"{i}. {claim.text[:100]}...")
        
        return "\n".join(lines)
    
    def format_for_synthesis(self, evidence: Evidence) -> str:
        """
        Format evidence for LLM synthesis.
        
        Args:
            evidence: Evidence object
            
        Returns:
            Formatted evidence string
        """
        return prompts.format_evidence_for_synthesis(
            claims=[{"claim": c.text, "confidence": c.confidence} for c in evidence.claims],
            documents=[{
                "title": d.title,
                "url": str(d.url),
                "snippet": d.snippet or d.content[:200]
            } for d in evidence.supporting_documents],
            relationships=[
                {
                    "subject": r[0].subject,
                    "predicate": r[0].predicate,
                    "object": r[0].object
                } for r in (evidence.relationship_chains or [])[:5]
            ] if evidence.relationship_chains else None
        )
    
    def enrich_with_graph_context(
        self,
        evidence: Evidence,
        subgraph: Subgraph
    ) -> Evidence:
        """
        Enrich evidence with additional graph context.
        
        Args:
            evidence: Existing evidence
            subgraph: Graph subgraph
            
        Returns:
            Enriched evidence
        """
        # Add entity information to claims
        entity_texts = {e.text for e in subgraph.entities}
        
        for claim in evidence.claims:
            # Check if claim mentions any graph entities
            mentioned_entities = [
                e for e in entity_texts
                if e.lower() in claim.text.lower()
            ]
            
            if mentioned_entities:
                if 'metadata' not in claim.__dict__:
                    claim.metadata = {}
                claim.metadata['mentioned_entities'] = mentioned_entities
        
        # Update relationship chains if not already present
        if not evidence.relationship_chains and subgraph.relationships:
            evidence.relationship_chains = self._extract_relationship_chains(subgraph)
        
        return evidence